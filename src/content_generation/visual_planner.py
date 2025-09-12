"""Visual Planner

Generates a machine-readable visual plan (beats, entities, shot types) from a
full narration script. Phase 3 stub: safe to import; not wired into pipeline yet.

Behavior:
- Uses local LLM (Ollama-compatible) or OpenAI per config.llm
- Emits JSON per schemas/visual_plan_v1.json
- Validates JSON if jsonschema is available (optional)
- Parses into Pydantic VisualPlan for downstream safety
- Persists the raw plan to output for audit
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover
    openai = None  # Lazy import environment

try:
    import jsonschema  # type: ignore
    from jsonschema import validate as jsonschema_validate
except Exception:  # pragma: no cover
    jsonschema = None
    jsonschema_validate = None

from .content_models import VisualPlan


class VisualPlanner:
    """LLM-backed visual planner that returns a validated VisualPlan."""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('video_ai.visual_planner')

        # Initialize LLM client (mirrors ScriptGenerator convention)
        self.use_local_llm = config.llm.use_local_llm
        if openai is None:
            raise RuntimeError("openai client not available in environment")

        if self.use_local_llm:
            self.client = openai.OpenAI(
                base_url=config.llm.local_llm_url,
                api_key="not-needed",
            )
            self.model_name = config.visual_planner.model_name
        else:
            # Use OpenAI API via centralized helper
            try:
                from src.llm.openai_client import get_openai_client, choose_model
                self.client = get_openai_client()
                self.model_name = choose_model("planner") or config.visual_planner.model_name
            except Exception:
                import os
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY is not set; please add it to .env.local")
                self.client = openai.OpenAI(api_key=api_key)
                self.model_name = os.environ.get("PLANNER_OPENAI_MODEL", config.visual_planner.model_name)

        # Paths
        self.project_root = Path(__file__).resolve().parents[2]
        self.schema_path = self.project_root / "schemas" / "visual_plan_v1.json"
        self.prompt_templates_image = self.project_root / "configs" / "prompt_templates_image.yaml"
        self.output_dir = Path(getattr(config.paths, 'output', './output')) / 'content_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load schema (optional)
        self._schema: Optional[Dict[str, Any]] = None
        if self.schema_path.exists():
            try:
                self._schema = json.loads(self.schema_path.read_text(encoding='utf-8'))
            except Exception as e:  # pragma: no cover
                self.logger.warning(f"Failed to load visual plan schema: {e}")

    # ------------------------ Public API ------------------------
    def plan_visuals(self, *, script_text: str, topic: str) -> VisualPlan:
        """Generate a visual plan from full narration text.

        Returns a Pydantic VisualPlan. Raises on fatal validation error.
        """
        style_template = self._get_style_template(topic)
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(script_text, topic, style_template)

        plan: VisualPlan
        try:
            response_text = self._call_llm(system_prompt, user_prompt)
            raw_json = self._extract_json(response_text)
            # Normalize to contract before Pydantic
            raw_json = self._coerce_plan(raw_json, topic, style_template, script_text)
            # Dump raw response for diagnostics
            try:
                diag_dir = Path(getattr(self.config.paths, 'output', './output')) / 'artifacts' / 'planner_debug'
                diag_dir.mkdir(parents=True, exist_ok=True)
                (diag_dir / f"raw_{topic}_{Path(self.schema_path).stem}_{Path(self.prompt_templates_image).stem}.txt").write_text(response_text, encoding='utf-8')
            except Exception:
                pass

            # Validate (optional)
            if self._schema and jsonschema_validate:
                try:
                    jsonschema_validate(instance=raw_json, schema=self._schema)
                except Exception as e:
                    self.logger.warning(f"Visual plan JSON schema validation failed: {e}")
                    # Non-fatal at this stage; continue to Pydantic parsing

            # Pydantic parse for type safety
            plan = VisualPlan(**raw_json)
        except Exception as e:
            # Robust fallback: build a minimal, valid plan from the script
            self.logger.warning(f"Visual planner failed to produce a valid plan, using fallback. Reason: {e}")
            plan = self._build_fallback_plan(script_text=script_text, topic=topic, style_template=style_template)

        # Persist artifact for audit
        out_path = self.output_dir / f"{topic}_visual_plan.json"
        try:
            out_path.write_text(json.dumps(plan.model_dump(), indent=2, ensure_ascii=False), encoding='utf-8')
            self.logger.info(f"Visual plan saved: {out_path}")
        except Exception as e:  # pragma: no cover
            self.logger.warning(f"Failed to save visual plan: {e}")

        return plan

    # ------------------------ Internals ------------------------
    def _build_system_prompt(self) -> str:
        vp = self.config.visual_planner
        shot_list = ", ".join(vp.shot_types)
        constraints = "\n- ".join(["Constraints:"] + vp.constraints) if vp.constraints else ""
        contract = (
            '{\n'
            '  "schema_version": "v1",\n'
            '  "global_style": {\n'
            '    "tone": "string", "color_palette": "string",\n'
            '    "camera_language": "string", "aspect_ratio": "string",\n'
            '    "topic": "string", "style_template_key": "string"\n'
            '  },\n'
            '  "entities": [{"id": "string", "kind": "string", "descriptor": "string"}],\n'
            '  "beats": [{\n'
            '    "id": "string", "title": "string", "summary": "string",\n'
            '    "estimated_duration_s": 0, "shot_type": "string",\n'
            '    "narration_span": {"start_token": 0, "end_token": 0},\n'
            '    "prompts": [{"type": "image", "prompt": "string"}]\n'
            '  }]\n'
            '}'
        )
        return (
            "You are a visual planner for a documentary-style YouTube video.\n"
            "Return ONLY a single JSON object matching this minimal contract.\n\n"
            f"CONTRACT (keys and types must exist):\n{contract}\n\n"
            f"Allowed shot types: {shot_list}.\n"
            "Use concise, filmable visuals (shots, b-roll). 6–14 beats recommended.\n"
            f"{constraints}\n"
            "Output: JSON only. No markdown."
        )

    def _build_user_prompt(self, script_text: str, topic: str, style_template: Dict[str, str]) -> str:
        # Derive dynamic inputs
        style_json = json.dumps(style_template, ensure_ascii=False)
        config = self.config.visual_planner
        adapters = getattr(self.config, 'topic_adapters', {}) or {}
        adapter = adapters.get(topic, adapters.get('default', {}))
        adapter_json = json.dumps(adapter, ensure_ascii=False)
        title = (script_text.splitlines()[0] or topic).strip()[:120]
        logline = " ".join(script_text.split()[:40]) + ("…" if len(script_text.split()) > 40 else "")
        # Simple beats outline by paragraphs
        paras = [p.strip() for p in script_text.split("\n\n") if p.strip()]
        outline = []
        for i, p in enumerate(paras[:10]):
            words = p.split()
            outline.append({
                "id": f"b{i+1:02d}",
                "title": (" ".join(words[:8]) + ("…" if len(words) > 8 else "")) or f"Section {i+1}",
                "summary": (" ".join(words[:28]) + ("…" if len(words) > 28 else "")),
                "approx_words": len(words)
            })
        payload = {
            "topic": topic,
            "title": title,
            "logline": logline,
            "style_template": style_template,
            "beat_target_seconds": config.beat_target_seconds,
            "adapter": adapter,
            "output_schema_version": config.output_schema_version,
            "beats_outline": outline,
            "narration": script_text,
        }
        return json.dumps(payload, ensure_ascii=False)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        vp = self.config.visual_planner
        # Try JSON mode if available (OpenAI)
        kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": vp.max_tokens,
            "temperature": vp.temperature,
        }
        # Always request JSON object mode for OpenAI-compatible clients
        kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        if not content:
            raise RuntimeError("Visual planner LLM returned empty content")
        # Log small preview for debugging
        preview = content[:400].replace("\n", " ")
        self.logger.info(f"Visual planner response preview: {preview}...")
        return content

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract a JSON object from LLM output (robust to minor wrappers)."""
        text = text.strip()
        # Fast path: starts with { and ends with }
        if text.startswith('{') and text.endswith('}'):
            return json.loads(text)
        # Fallback: find first { ... } block
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("Could not extract JSON object from planner output")

    def _coerce_plan(self, data: Dict[str, Any], topic: str, style_template: Dict[str, str], script_text: str) -> Dict[str, Any]:
        # Unwrap common wrapper
        if isinstance(data, dict) and "visual_plan" in data and isinstance(data["visual_plan"], dict):
            data = data["visual_plan"]
        # Global style defaults
        gs = data.get("global_style") or {}
        if not isinstance(gs, dict):
            gs = {}
        gs.setdefault("tone", "documentary")
        gs.setdefault("color_palette", "neutral_warm")
        gs.setdefault("camera_language", "wide_establishers, slow_pushins, cutaways")
        gs.setdefault("aspect_ratio", "16:9")
        # Required keys for our GlobalStyle model
        gs.setdefault("topic", topic)
        gs.setdefault("style_template_key", topic)
        data["global_style"] = gs
        data.setdefault("schema_version", "v1")
        # Entities normalization
        entities = data.get("entities") or []
        norm_entities = []
        if isinstance(entities, list):
            for e in entities:
                if not isinstance(e, dict):
                    continue
                eid = e.get("id") or e.get("name") or e.get("title") or "Entity"
                kind = e.get("kind") or "concept"
                desc = e.get("descriptor") or e.get("description") or eid
                norm_entities.append({"id": str(eid), "kind": str(kind), "descriptor": str(desc)})
        data["entities"] = norm_entities
        # Beats normalization
        beats = data.get("beats") or data.get("scenes") or data.get("sections") or []
        if not isinstance(beats, list):
            beats = []
        norm_beats = []
        # Estimate total seconds from script
        words = script_text.split()
        target_total_s = max(60.0, len(words) / float(getattr(self.config, 'planner', {}).get('target_wpm', 150)) * 60.0) if isinstance(getattr(self.config, 'planner', {}), dict) else max(60.0, len(words)/150*60)
        if not beats:
            # Build simple beats if missing
            paras = [p.strip() for p in script_text.split("\n\n") if p.strip()]
            for i, p in enumerate(paras[:8]):
                norm_beats.append({
                    "id": f"beat_{i+1:03d}",
                    "title": (p.split(". ")[0][:80] if p else f"Section {i+1}"),
                    "summary": (" ".join(p.split()[:28]) + ("…" if len(p.split())>28 else "")),
                    "estimated_duration_s": 10.0,
                    "visuals": ["establishing wide", "insert detail"],
                })
        else:
            for i, b in enumerate(beats):
                if not isinstance(b, dict):
                    continue
                bid = b.get("id") or f"beat_{i+1:03d}"
                title = b.get("title") or b.get("heading") or f"Beat {i+1}"
                summary = b.get("summary") or b.get("desc") or ""
                dur = b.get("estimated_duration_s") or b.get("duration_s") or 10.0
                visuals = b.get("visuals") or b.get("shots") or []
                if isinstance(visuals, str):
                    visuals = [visuals]
                visuals = [str(v)[:120] for v in visuals][:4]
                norm_beats.append({
                    "id": str(bid),
                    "title": str(title),
                    "summary": str(summary),
                    "estimated_duration_s": float(dur),
                    "visuals": visuals or ["establishing"],
                })
        # Scale durations to target ±10%
        total = sum(b.get("estimated_duration_s", 0.0) for b in norm_beats) or 1.0
        scale = target_total_s / total
        if abs(scale - 1.0) > 0.1:
            for b in norm_beats:
                b["estimated_duration_s"] = float(max(6.0, min(180.0, b.get("estimated_duration_s", 10.0) * scale)))
        # Enforce required fields per beat
        shot_types = list(getattr(self.config.visual_planner, 'shot_types', [])) or [
            "establishing","medium_detail","insert","map","diagram","archival","reenactment","abstract"
        ]
        total_secs = sum(b.get("estimated_duration_s", 0.0) for b in norm_beats) or 1.0
        tokens = script_text.split()
        total_tokens = len(tokens) or 1
        cursor = 0
        fixed_beats = []
        for i, b in enumerate(norm_beats):
            dur = float(b.get("estimated_duration_s", 10.0))
            portion = max(1, int(round((dur / total_secs) * total_tokens)))
            start_tok = int(cursor)
            end_tok = int(min(total_tokens, cursor + portion))
            # Clamp strictly in-range and increasing
            start_tok = max(0, min(total_tokens, start_tok))
            end_tok = max(start_tok, min(total_tokens, end_tok))
            cursor = end_tok

            visuals = b.get("visuals") or ["establishing"]
            if isinstance(visuals, str):
                visuals = [visuals]
            prompt_text = (visuals[0] if visuals else b.get("title", ""))
            prompt_text = str(prompt_text)[:180]
            prompts = b.get("prompts")
            if not isinstance(prompts, list) or not prompts:
                prompts = [{
                    "type": "image",
                    "prompt": prompt_text,
                    "negatives": "logos, watermarks, celebrity likeness",
                    "style_locks": [gs.get("tone","documentary"), gs.get("color_palette","neutral_warm")],
                    "variation": 0.0,
                    "seed_group": b.get("id", f"beat_{i+1:03d}")
                }]

            shot = b.get("shot_type")
            if not shot or shot not in shot_types:
                shot = shot_types[i % len(shot_types)]

            fixed_beats.append({
                "id": b.get("id", f"beat_{i+1:03d}"),
                "title": b.get("title", f"Beat {i+1}"),
                "summary": b.get("summary", ""),
                "estimated_duration_s": dur,
                "shot_type": shot,
                "seed_group": b.get("id", f"beat_{i+1:03d}"),
                "narration_span": {"start_token": start_tok, "end_token": end_tok},
                "prompts": prompts,
                "overlays": b.get("overlays", []),
                "notes": b.get("notes", "")
            })
        if cursor < total_tokens and fixed_beats:
            fixed_beats[-1]["narration_span"]["end_token"] = total_tokens
        data["beats"] = fixed_beats
        # Audit log
        try:
            self.logger.info(f"Planner audit: beats={len(norm_beats)} total_s={sum(b['estimated_duration_s'] for b in norm_beats):.1f} entities={len(norm_entities)}")
        except Exception:
            pass
        return data

    def _get_style_template(self, topic: str) -> Dict[str, str]:
        tpl = self.config.image_generation.get_style_for_topic(topic)
        return {
            "base_style": tpl.base_style,
            "colors": tpl.colors,
            "mood": tpl.mood,
        }

    # ------------------------ Fallback Builder ------------------------
    def _build_fallback_plan(self, *, script_text: str, topic: str, style_template: Dict[str, str]) -> VisualPlan:
        """Construct a minimal, schema-valid plan from the script when LLM output is invalid."""
        words = script_text.split()
        total_words = max(1, len(words))
        # Estimate seconds at ~2.5 words/sec (150 wpm)
        est_total_s = total_words / 2.5
        target = getattr(self.config.visual_planner, 'beat_target_seconds', [6, 15])
        avg_len = (target[0] + target[1]) / 2 if isinstance(target, list) and len(target) == 2 else 10.0
        num_beats = max(1, int(round(est_total_s / max(6.0, min(20.0, avg_len)))))
        num_beats = min(max(3, num_beats), 8)  # keep it sane for short scripts

        tokens_per_beat = max(1, total_words // num_beats)
        shot_types = list(getattr(self.config.visual_planner, 'shot_types', [
            "establishing", "medium_detail", "insert", "diagram", "map"
        ]))

        beats = []
        start_idx = 0
        for i in range(num_beats):
            end_idx = total_words if i == num_beats - 1 else min(total_words, start_idx + tokens_per_beat)
            snippet = " ".join(words[start_idx:end_idx])
            shot = shot_types[i % len(shot_types)] if shot_types else "establishing"
            beat_id = f"beat_{i+1:03d}"
            beats.append({
                "id": beat_id,
                "narration_span": {"start_token": int(start_idx), "end_token": int(end_idx)},
                "estimated_duration_s": float(max(target[0], min(target[1], avg_len)) if isinstance(target, list) and len(target) == 2 else 10.0),
                "shot_type": shot,
                "seed_group": beat_id,
                "prompts": [
                    {
                        "type": "image",
                        "prompt": f"{snippet[:180]}" if snippet else topic,
                        "negatives": "logos, watermarks, celebrity likeness",
                        "style_locks": [style_template.get("base_style", "cinematic"), style_template.get("mood", "documentary")],
                        "variation": 0.0,
                        "seed_group": beat_id
                    }
                ],
                "overlays": [],
                "notes": "fallback auto-planned"
            })
            start_idx = end_idx

        plan_dict = {
            "schema_version": "v1",
            "global_style": {
                "topic": topic,
                "style_template_key": topic,
                "aspect_ratio": "16:9",
                "color_profile": "sRGB",
                "tone": "documentary"
            },
            "entities": [],
            "beats": beats,
        }
        return VisualPlan(**plan_dict)


