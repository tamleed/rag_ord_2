# Transfer checklist: this repo -> other PC -> server

## 1) Push branch from Codex environment (if remote configured)

```bash
git status -sb
git branch --show-current
# optional
git remote -v
# push current branch
git push -u origin codex/handoff-transfer-instructions
```

If remote is not configured in this environment, do steps below on your main workstation where `origin` exists.

## 2) On your other computer: get the exact branch

```bash
git fetch --all --prune
git checkout -b codex/handoff-transfer-instructions origin/codex/handoff-transfer-instructions
git status -sb
git log --oneline -n 5
```

## 3) Prepare archive from that branch (other computer)

PowerShell example:

```powershell
Compress-Archive -Path .\* -DestinationPath ..\rag_ord_2_v2.zip -Force
```

## 4) Copy archive to server

PowerShell + scp:

```powershell
scp ..\rag_ord_2_v2.zip ubuntu@<SERVER_IP>:/home/ubuntu/
```

## 5) Unpack on server into new directory

```bash
cd /home/ubuntu
mkdir -p faq_rag_v2
unzip -o rag_ord_2_v2.zip -d faq_rag_v2
cd faq_rag_v2
ls -la
```

## 6) Run v2 in parallel (do not touch old service)

- keep old service as-is;
- run new project on another port (example: `8010`);
- route `/answer_v2` and `/answer_full_v2` to new backend in nginx.
