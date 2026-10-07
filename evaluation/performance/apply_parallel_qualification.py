from __future__ import annotations

from pathlib import Path


TARGET = Path("mvp/src/generative_v4/qualification_chain.py")

IMPORT_OLD = "from __future__ import annotations\n\nimport re\n"
IMPORT_NEW = "from __future__ import annotations\n\nfrom concurrent.futures import ThreadPoolExecutor\nimport re\n"

CONSTANT_OLD = (
    "SEMANTIC_QUALIFICATION_SHORTLIST_SIZE = 48\n"
    "SEMANTIC_QUALIFICATION_BATCH_SIZE = 6\n"
)
CONSTANT_NEW = (
    "SEMANTIC_QUALIFICATION_SHORTLIST_SIZE = 48\n"
    "SEMANTIC_QUALIFICATION_BATCH_SIZE = 6\n"
    "SEMANTIC_QUALIFICATION_MAX_WORKERS = 4\n"
)

LOOP_START = "    base_frame = candidate_subset.copy()\n    for start in range(0, len(candidate_subset), SEMANTIC_QUALIFICATION_BATCH_SIZE):\n"
LOOP_END = "\n    qualified_frame = pd.DataFrame(qualified_rows)\n"

REPLACEMENT = '''    base_frame = candidate_subset.copy()\n\n    # Each qualification batch is independent: same request, disjoint candidate rows,\n    # no shared ranking state. Running these calls concurrently preserves the exact\n    # candidate set, prompt, model, batch size, and downstream ordering while removing\n    # avoidable network wait time. Keep concurrency deliberately bounded to reduce\n    # rate-limit risk.\n    batch_jobs: list[tuple[int, pd.DataFrame, list[dict[str, Any]], list[str]]] = []\n    for batch_index, start in enumerate(\n        range(0, len(candidate_subset), SEMANTIC_QUALIFICATION_BATCH_SIZE)\n    ):\n        batch = candidate_subset.iloc[\n            start : start + SEMANTIC_QUALIFICATION_BATCH_SIZE\n        ].copy().reset_index(drop=True)\n        candidate_payloads = [\n            _candidate_payload_for_llm(_normalize_candidate_payload(row))\n            for _, row in batch.iterrows()\n        ]\n        candidate_ids = [\n            str(payload.get("candidate_id") or payload.get("source_id") or "")\n            for payload in candidate_payloads\n        ]\n        duplicate_ids = sorted(\n            {candidate_id for candidate_id in candidate_ids if candidate_ids.count(candidate_id) > 1}\n        )\n        if duplicate_ids:\n            raise RuntimeError(\n                f"Qualification batch contains duplicate candidate IDs: {duplicate_ids}"\n            )\n        batch_jobs.append((batch_index, batch, candidate_payloads, candidate_ids))\n\n    def _run_batch(job):\n        batch_index, batch, candidate_payloads, candidate_ids = job\n        llm_outputs, llm_call_count, input_tokens, output_tokens, batch_total_tokens = (\n            _qualify_with_llm(request, candidate_payloads)\n        )\n        return (\n            batch_index,\n            batch,\n            candidate_ids,\n            llm_outputs,\n            llm_call_count,\n            input_tokens,\n            output_tokens,\n            batch_total_tokens,\n        )\n\n    max_workers = min(SEMANTIC_QUALIFICATION_MAX_WORKERS, len(batch_jobs))\n    if max_workers <= 1:\n        batch_results = [_run_batch(job) for job in batch_jobs]\n    else:\n        with ThreadPoolExecutor(max_workers=max_workers) as executor:\n            # executor.map preserves input order; sorting again below is defensive and\n            # makes the ordering guarantee explicit for downstream record construction.\n            batch_results = list(executor.map(_run_batch, batch_jobs))\n\n    batch_results.sort(key=lambda item: item[0])\n\n    for (\n        _,\n        batch,\n        candidate_ids,\n        llm_outputs,\n        llm_call_count,\n        input_tokens,\n        output_tokens,\n        batch_total_tokens,\n    ) in batch_results:\n        total_llm_call_count += int(llm_call_count or 0)\n        if total_input_tokens is None:\n            total_input_tokens = input_tokens\n        elif input_tokens is not None:\n            total_input_tokens += int(input_tokens)\n        if total_output_tokens is None:\n            total_output_tokens = output_tokens\n        elif output_tokens is not None:\n            total_output_tokens += int(output_tokens)\n        if total_tokens is None:\n            total_tokens = batch_total_tokens\n        elif batch_total_tokens is not None:\n            total_tokens += int(batch_total_tokens)\n\n        llm_used = llm_outputs is not None\n        output_lookup: dict[str, QualificationOutput] = {}\n        if llm_outputs is not None:\n            parsed_ids = [str(item.candidate_id) for item in llm_outputs]\n            parsed_lookup = {str(item.candidate_id): item for item in llm_outputs}\n            duplicate_parsed_ids = sorted(\n                {candidate_id for candidate_id in parsed_ids if parsed_ids.count(candidate_id) > 1}\n            )\n            missing_ids = sorted(set(candidate_ids).difference(parsed_lookup))\n            extra_ids = sorted(set(parsed_lookup).difference(candidate_ids))\n            if duplicate_parsed_ids or missing_ids or extra_ids:\n                raise RuntimeError(\n                    "Qualification LLM returned mismatched candidate IDs: "\n                    f"duplicates={duplicate_parsed_ids}, missing={missing_ids}, extra={extra_ids}"\n                )\n            output_lookup = parsed_lookup\n\n        for _, row in batch.iterrows():\n            candidate_payload = _normalize_candidate_payload(row)\n            if candidate_payload.get("predicted_preference") is None:\n                candidate_payload["predicted_preference"] = 0.0\n            if candidate_payload.get("raw_rank") is None:\n                candidate_payload["raw_rank"] = 0\n            if candidate_payload.get("request_relevance_score") is None:\n                candidate_payload["request_relevance_score"] = float(\n                    candidate_payload.get("request_relevance") or 0.0\n                )\n            if candidate_payload.get("request_relevance_rank") is None:\n                candidate_payload["request_relevance_rank"] = 0\n            candidate = RetrievedCandidate.model_validate(candidate_payload)\n            output = output_lookup.get(candidate.candidate_id) if output_lookup else None\n            if output is None:\n                output = _fallback_qualification_output(request, candidate)\n            structured_ok, structured_supported, structured_unsupported = (\n                _structured_constraint_satisfied(candidate, spec.structured_constraints)\n            )\n            if (\n                not structured_ok or output.violated_semantic_exclusions\n            ) and output.status != "unsupported":\n                violated = list(dict.fromkeys(output.violated_semantic_exclusions))\n                if not structured_ok:\n                    violated.extend(structured_unsupported)\n                output = QualificationOutput(\n                    candidate_id=output.candidate_id,\n                    status="unsupported",\n                    supported_required_aspects=list(\n                        dict.fromkeys(\n                            [*output.supported_required_aspects, *structured_supported]\n                        )\n                    ),\n                    unsupported_required_aspects=list(\n                        dict.fromkeys(\n                            [*output.unsupported_required_aspects, *structured_unsupported]\n                        )\n                    ),\n                    supported_preferred_aspects=output.supported_preferred_aspects,\n                    unsupported_preferred_aspects=output.unsupported_preferred_aspects,\n                    violated_semantic_exclusions=list(dict.fromkeys(violated)),\n                    grounded_evidence=output.grounded_evidence,\n                    grounded_evidence_details=output.grounded_evidence_details,\n                    qualification_reason="Candidate violates a structured hard constraint.",\n                    request_match=None,\n                    caveat=(\n                        f"Unsupported structured constraints: {', '.join((structured_unsupported or structured_supported)[:5])}"\n                        if (structured_supported or structured_unsupported)\n                        else "Candidate violates a structured hard constraint."\n                    ),\n                )\n            record = _record_from_output(\n                candidate.candidate_id, output, llm_used=llm_used\n            )\n            records.append(record)\n            qualified_row = dict(candidate_payload)\n            qualified_row.update(record.model_dump())\n            qualified_row["candidate_id"] = candidate.candidate_id\n            qualified_row["source_id"] = candidate.candidate_id\n            if record.qualification_status != "unsupported":\n                qualified_rows.append(qualified_row)\n'''


def main() -> None:
    text = TARGET.read_text()

    if "SEMANTIC_QUALIFICATION_MAX_WORKERS = 4" in text:
        print("Parallel qualification patch is already applied.")
        return

    if IMPORT_OLD not in text:
        raise RuntimeError("Expected import anchor was not found; refusing to patch.")
    text = text.replace(IMPORT_OLD, IMPORT_NEW, 1)

    if CONSTANT_OLD not in text:
        raise RuntimeError("Expected qualification constants were not found; refusing to patch.")
    text = text.replace(CONSTANT_OLD, CONSTANT_NEW, 1)

    start = text.find(LOOP_START)
    end = text.find(LOOP_END, start)
    if start < 0 or end < 0:
        raise RuntimeError("Expected qualification loop was not found; refusing to patch.")

    text = text[:start] + REPLACEMENT + text[end:]
    TARGET.write_text(text)
    compile(text, str(TARGET), "exec")
    print(f"Applied bounded parallel qualification to {TARGET}.")
    print("Concurrency: max 4 workers; candidate set, batch size, prompts, model, and ordering unchanged.")


if __name__ == "__main__":
    main()
