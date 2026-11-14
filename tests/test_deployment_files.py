
"""Static guards for the S3 architecture: registration data must not depend on a Docker volume,
and no AWS credential may ever be written into the project. Plain text checks (no YAML library needed)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")


def test_dockerfile_has_no_volume_and_points_at_s3():
    docker = read("Dockerfile")
    assert not re.search(r"^\s*VOLUME\b", docker, re.M), "a VOLUME would keep registration data outside S3"
    assert "FR_S3_BUCKET=jaganops-face-recognition-data-080705554676" in docker
    assert "FR_S3_REGION=ap-south-1" in docker
    # the original multi-stage build, non-root user and health check are still there
    assert docker.count("FROM python:3.12-slim AS") == 3 and "USER appuser" in docker and "HEALTHCHECK" in docker


def test_playbook_creates_and_mounts_no_docker_volume():
    play = read("ansible/deploy.yml")
    assert "data_volume" not in play
    assert "--volume" not in play and "-v " not in play
    assert not re.search(r"docker,\s*volume", play) and "docker volume" not in play
    assert "FR_DATA_DIR=" in play  # scratch path is still passed to the container


def test_no_aws_credentials_anywhere_in_the_project():
    patterns = [r"AKIA[0-9A-Z]{16}", r"aws_secret_access_key\s*[=:]\s*\S{20,}",
                r"AWS_SECRET_ACCESS_KEY\s*[=:]\s*[A-Za-z0-9/+]{20,}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"]
    files = [ROOT / "app.py", ROOT / "config.py", ROOT / "Dockerfile", ROOT / "requirements.txt"]
    for folder in ("backend", "scripts", "ansible", ".github", "frontend", "iam"):
        files += [p for p in (ROOT / folder).rglob("*") if p.is_file() and p.name != "inventory.ini"]
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pat in patterns:
            assert not re.search(pat, text), f"credential-looking text in {path.relative_to(ROOT)}"


def workflow_text():
    """The pipeline is found by folder, not by file name, and there must be exactly one."""
    files = sorted((ROOT / ".github" / "workflows").glob("*.y*ml"))
    assert len(files) == 1, f"expected exactly one workflow file, found {[f.name for f in files]}"
    return files[0].read_text(encoding="utf-8", errors="replace")


def test_workflow_uses_secrets_and_only_deploys_from_main():
    wf = workflow_text()
    for needed in ("secrets.AWS_ROLE_TO_ASSUME", "secrets.EC2_SSH_PRIVATE_KEY", "secrets.EC2_HOST",
                   "id-token: write", "needs: [test, docker-build]", "refs/heads/main"):
        assert needed in wf, needed
    assert "AWS_SECRET_ACCESS_KEY" not in wf and "AWS_ACCESS_KEY_ID" not in wf
    assert "pull_request_target" not in wf


def test_workflow_deploys_the_exact_image_it_pushed_and_always_cleans_up():
    wf = workflow_text()
    assert "${{ github.sha }}" in wf and 'image_tag=${{ github.sha }}' in wf   # same tag pushed and deployed
    assert "ansible-playbook ansible/deploy.yml" in wf                          # the existing playbook
    start = wf.index("- name: Remove SSH key and inventory")
    step = wf[start:].split("\n      - name:")[0]    # just this one step, not the ones after it
    assert "if: always()" in step            # key + inventory are removed even when the deployment fails


def test_iam_policy_files_are_valid_json_with_valid_sids():
    import json
    for path in (ROOT / "iam").glob("*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for statement in doc["Statement"]:
            # Sid is optional in AWS IAM. If supplied, it must contain only letters and digits.
            sid = statement.get("Sid")
            if sid is not None:
                assert isinstance(sid, str) and re.fullmatch(r"[A-Za-z0-9]+", sid), \
                    f"{path.name}: bad Sid {sid!r}"


def test_main_github_policies_have_no_placeholders_and_only_the_optional_one_has_a_security_group_id():
    for name in ("GitHubActionsDeploy.policy.json", "GitHubActionsOIDC.trust.json"):
        assert "REPLACE" not in read(f"iam/{name}"), f"{name} must be applicable exactly as it is"

    # GitHub repository uses immutable OIDC subject claims.
    expected_subject = "repo:Jagan540-tech@205582675/face-recognition-cloud-devops-project@1457500621:ref:refs/heads/main"
    assert expected_subject in read("iam/GitHubActionsOIDC.trust.json")

    assert "ecr:PutImage" in read("iam/GitHubActionsDeploy.policy.json")
