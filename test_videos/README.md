# test_videos

Drop short video clips here to run OCP Sentry without any camera. The files
themselves are git-ignored (large, and not ours to redistribute).

`config/cameras.testvideos.example.yaml` expects these names:

| File | Used by | Suggested content |
|------|---------|-------------------|
| `VIRAT_S_000002.mp4` | `cam1` | people walking — e.g. a clip from the [VIRAT Video Dataset](https://viratdata.org/) |
| `fire2.mp4` | `cam2` | visible fire and/or smoke |
| `nofire1.mp4` | `cam3` | a scene with no fire (checks for false positives) |

Use any clips you like — just keep the file names in sync with your
`config/cameras.yaml`. Then start the stack with the override file:

```bash
docker compose -f docker-compose.yml -f docker-compose.testvideos.yml up -d
```
