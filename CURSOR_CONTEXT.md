Long Video AI Automation – Context Reference

Reference token: LVAI-CONTEXT-V1-RTX5080-OAI-PLANNER-QA

Purpose
- One-file overview to quickly restore project context in new sessions. Mention the token above to locate this summary.

High-level architecture
- Entry: `main.py` orchestrates content → media → assembly. Modes: interactive, single, auto. `--test` flips short, fast settings.
- Config: `configs/config.yaml` centralizes all knobs (content, llm, tts, image_generation, visual_planner, continuity, alignment, video, performance, paths, logging, debug).
- Pipelines:
  - Content: `src/content_generation/` → research → script generation (OpenAI) → visual planning (OpenAI) → alignment (heuristic/Aeneas-ready) → enhanced prompts build → artifacts.
  - Media: `src/media_generation/` → TTS (SAPI default), image generation (SDXL), QA + retries, artifacts and timeline emission.
  - Assembly: `src/video_assembly/` → FFmpeg/NVENC assembly, uses `timeline.json` when available.
- Automation: `src/automation/` manages schedules and queue.

Key features (current)
- LLM provider: OpenAI `gpt-4o-mini` for both story and visual planner with JSON responses.
- Chunked script gen: outline + per-beat expansion; artifacts in `output/artifacts/script_debug/`.
- Visual Planner: beat-level shot list; coercion to schema; artifacts per run.
- Deterministic seeding: entity continuity; salted seeds on fallback.
- QA loop: BLIP captioning + similarity (embeddings or token overlap), topic-aware thresholds, bad caption auto-retry.
- Test mode `--test`: 1–2 minute fast path, SAPI TTS, reduced SDXL steps.
- RTX 5080 strict: CUDA + device name validated; NVENC video encoding.
- Possessive normalization for s-ending names for better TTS.

Important paths and artifacts
- Output video: `output/videos/*.mp4`
- Artifacts per run: `output/artifacts/<run_id>/` → `visual_plan.json`, `alignment.json`, `prompts.json/csv`, `qa_report.json`, `timeline.json`
- Script debug: `output/artifacts/script_debug/` → `outline_*.json`, `beats_*.json`, `narration_*.txt`, `full_scripts/`
- Content results (JSON bundle): `output/content_results/*.json`

CLI usage
- Test single: `python main.py --test --mode single --topic mythology --subtopic Zeus`
- Prod single: `python main.py --mode single --topic space --subtopic Europa`
- Interactive: `python main.py --mode interactive`
- Automated: `python main.py --mode auto`

Config highlights (`configs/config.yaml`)
- `llm`: `use_local_llm: false`, `model_name: gpt-4o-mini`
- `tts`: `engine: "sapi"`
- `image_generation`: SDXL tuned defaults; templates for mythology/history to avoid statue bias
- `visual_planner`: `enabled: true`, `use_enhanced_prompts: true`, retries/fallbacks
- `continuity`: `caption_similarity_threshold` base (topic overrides in code)
- `alignment`: `captioner: blip`, `source: heuristic`
- `video`: `codec: h264_nvenc`, `ken_burns_effect: false`
- `debug`: `save_full_scripts: true`

Directory map (purpose)
- `configs/` – YAML config, topics examples
- `data/` – topic files; queue/completed topics
- `output/` – audio, images, videos, content_results, artifacts
- `src/automation/` – scheduler and models
- `src/content_generation/` – models, prompts, research engine, script generator, visual planner, alignment, topic queue
- `src/media_generation/` – image_generator, tts_engine, media_pipeline, image_prompt_builder, media_tester
- `src/video_assembly/` – video_assembler, audio_processor, video_effects, metadata_spoofer, video models
- `src/utils/` – config loader, captions, similarity, text_normalize, logger, seeds (if present)
- `models/` – tts/image model folders (if used locally)
- `temp/` – working files (audio, cache, images)

Key files (short purpose)
- `main.py` – entrypoint, CLI, progress bars, strict GPU checks, passes `timeline_path` to assembler
- `src/llm/openai_client.py` – OpenAI client init and model chooser
- `src/content_generation/script_generator.py` – outline+expand loop, artifacts, possessive normalization
- `src/content_generation/visual_planner.py` – planner call, output coercion, schema defaults
- `src/content_generation/alignment.py` – alignment helpers (heuristic/Aeneas-ready)
- `src/media_generation/image_prompt_builder.py` – prompt building with negatives, shot normalization
- `src/media_generation/media_pipeline.py` – orchestrates TTS+images; QA with retries; emits `qa_report.json`, `timeline.json`
- `src/media_generation/image_generator.py` – SDXL generation, prompt sanitization, presets
- `src/media_generation/tts_engine.py` – SAPI default, text cleaning, TTS chunking, concatenation
- `src/utils/captions.py` – BLIP captioning with normalization
- `src/utils/similarity.py` – embeddings/token overlap + topic-aware thresholds
- `src/utils/text_normalize.py` – name possessive normalization for s-ending names
Operational notes
- Windows environment; activate venv before running. OpenAI key in `.env.local`.
- GPU-only policy: if CUDA/RTX 5080 unavailable → exit.
- Ken Burns disabled by default per user preference.

