import json
import os
from glob import glob
from typing import Any, Dict, List, Optional


class CostReconciler:
    """Reconciles response costs with OpenRouter generation records."""

    def __init__(
        self,
        *,
        completion_api: Any,
        results_store: Any,
        generation_ids: List[str],
    ) -> None:
        self.completion_api = completion_api
        self.results_store = results_store
        self.generation_ids = generation_ids
        self._response_cost_cache: Dict[str, float] = {}

    def lookup_response_cost_from_raw(self, generation_id: Optional[str]) -> float:
        if not generation_id:
            return 0.0
        if generation_id in self._response_cost_cache:
            return self._response_cost_cache[generation_id]

        pattern = os.path.join("raw", f"*_{generation_id}.json")
        response_cost = 0.0
        for raw_path in glob(pattern):
            try:
                with open(raw_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue

            usage = ((payload.get("response_data") or {}).get("usage") or {})
            try:
                response_cost = float(usage.get("cost") or 0.0)
            except (TypeError, ValueError):
                response_cost = 0.0
            break

        self._response_cost_cache[generation_id] = response_cost
        return response_cost

    def update_costs_from_openrouter(self) -> None:
        print("\nRetrieving actual costs from OpenRouter...")

        openrouter_ids = list(self.generation_ids)
        if not openrouter_ids:
            print("No generation IDs to update costs for")
            return

        openrouter_ids_to_fetch = []
        seen_generation_ids = set()
        for _, model_results in self.results_store.data["results"].items():
            for result in model_results:
                gen_id = result.get("generation_id")
                if self._has_final_openrouter_cost(result):
                    continue
                if gen_id and gen_id not in seen_generation_ids:
                    seen_generation_ids.add(gen_id)
                    openrouter_ids_to_fetch.append(gen_id)

        if not openrouter_ids_to_fetch:
            print("No unresolved OpenRouter generation IDs found. Skipping cost fetch.")
            return

        if not self.completion_api.openrouter_api_key:
            print("OPENROUTER_API_KEY not set. Skipping OpenRouter cost fetch.")
            return

        costs = self.completion_api.get_openrouter_costs(openrouter_ids_to_fetch)
        total_cost_before = 0.0
        total_cost_after = 0.0
        generation_count = 0
        byok_response_count = 0
        estimated_count = 0
        unresolved_count = 0

        for _, model_results in self.results_store.data["results"].items():
            for result in model_results:
                old_cost = float(result.get("api_cost_usd", 0.0) or 0.0)
                total_cost_before += old_cost
                if self._has_final_openrouter_cost(result):
                    total_cost_after += old_cost
                    continue

                gen_id = result.get("generation_id")
                response_cost = result.get("response_api_cost_usd")
                if response_cost is None:
                    response_cost = self.lookup_response_cost_from_raw(gen_id)
                if not response_cost:
                    response_cost = result.get("api_cost_usd", 0.0)
                response_cost = float(response_cost or 0.0)
                result["response_api_cost_usd"] = response_cost

                new_cost = response_cost
                cost_source = "response" if response_cost > 0 else "pending"

                record = costs.get(gen_id) if gen_id else None
                if isinstance(record, dict):
                    record_cost = float(record.get("cost", 0.0) or 0.0)
                    is_byok = bool(record.get("is_byok"))
                    if is_byok:
                        new_cost = response_cost
                        cost_source = "response_byok" if response_cost > 0 else "pending_byok"
                        byok_response_count += 1
                    else:
                        new_cost = record_cost
                        cost_source = "openrouter_generation" if record_cost > 0 else "pending_generation"
                        generation_count += 1

                if new_cost == 0.0:
                    estimated_cost = self.completion_api.estimate_model_catalog_cost_usd(
                        result["model_name"],
                        int(result.get("input_tokens", 0) or 0),
                        int(result.get("output_tokens", 0) or 0),
                    )
                    if estimated_cost > 0.0:
                        new_cost = estimated_cost
                        cost_source = "model_catalog_estimate"
                        estimated_count += 1

                if new_cost == 0.0:
                    unresolved_count += 1
                    cost_source = "unresolved"

                result["api_cost_usd"] = new_cost
                result["cost_source"] = cost_source
                total_cost_after += new_cost

        total_cost_update = total_cost_after - total_cost_before
        self.results_store.data["metadata"]["total_api_cost_usd"] = total_cost_after
        self.completion_api.api_usage.total_cost_usd = total_cost_after

        self.results_store.save()
        print(
            f"Updated costs from OpenRouter. "
            f"Total cost adjustment: ${total_cost_update:.4f}"
        )
        print(
            f"Cost sources after reconciliation: generation={generation_count}, "
            f"byok_response={byok_response_count}, catalog_estimate={estimated_count}, "
            f"unresolved={unresolved_count}"
        )
        if unresolved_count:
            print(
                f"Warning: OpenRouter cost lookup did not resolve {unresolved_count} "
                f"result rows after all fallbacks."
            )

    @staticmethod
    def _has_final_openrouter_cost(result: Dict[str, Any]) -> bool:
        if not result.get("generation_id"):
            return False
        return str(result.get("cost_source") or "") in {
            "openrouter_generation",
            "response_byok",
        }
