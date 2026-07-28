# junupark.xyz

Personal Hugo site for Junu Park.

Public sections:

- `/` — home
- `/projects/` — personal projects and open-source work
- `/blog/` — non-technical essays
- `/llm-engine/` — interactive end-to-end LLM inference engine explainer

ML systems learning notes remain in
[`junuxyz/mlsys-notes`](https://github.com/junuxyz/mlsys-notes).

Run the Hugo site locally with `hugo server`.

The interactive LLM engine explainer is an isolated Vite app:

```bash
cd llm-engine
npm ci
npm run dev
```
