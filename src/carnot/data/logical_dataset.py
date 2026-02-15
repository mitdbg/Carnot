from __future__ import annotations


class LogicalDataset:
    def __init__(self, name: str, parents: list[LogicalDataset] | None = None, id_params: dict | None = None, **kwargs):
        self.name = name
        self.parents = parents or []
        self.id_params = id_params or {
            "limit_id": 0,
            "merge_id": 0,
            "code_id": 0,
            "reason_id": 0,
            "sem_agg_id": 0,
            "sem_filter_id": 0,
            "sem_map_id": 0,
            "sem_flat_map_id": 0,
            "sem_groupby_id": 0,
            "sem_join_id": 0,
            "sem_topk_id": 0,
        }
        self.params = kwargs
        self.output_dataset_id = kwargs.get("output_dataset_id") or name

    def serialize(self) -> dict:
        return {
            "name": self.name,
            "output_dataset_id": self.output_dataset_id,
            "params": self.params,
            "parents": [p.serialize() for p in self.parents],
        }

    def limit(self, n: int) -> LogicalDataset:
        """
        Apply a limit operation to the dataset, returning only the first n records.
        """
        limited_name = f"LimitOperation{self.id_params['limit_id'] + 1}"
        self.id_params["limit_id"] += 1
        params = {"operator": "Limit", "description": f"Limited {self.name} to first {n} records", "n": n}
        return LogicalDataset(limited_name, parents=[self], id_params=self.id_params, output_dataset_id=limited_name, **params)

    def merge(self, other: LogicalDataset) -> LogicalDataset:
        """
        Merge this dataset with another dataset.
        """
        merged_name = f"MergeOperation{self.id_params['merge_id'] + 1}"
        self.id_params["merge_id"] += 1
        params = {"operator": "Merge", "description": f"Merged {self.name} with {other.name}"}
        return LogicalDataset(merged_name, parents=[self, other], id_params=self.id_params, output_dataset_id=merged_name, **params)

    def code(self, task: str) -> LogicalDataset:
        """
        Apply a code operation to the dataset based on the given task.
        """
        coded_name = f"CodeOperation{self.id_params['code_id'] + 1}"
        self.id_params["code_id"] += 1
        params = {"operator": "Code", "description": f"Coded {self.name} for task: {task}", "task": task}
        return LogicalDataset(coded_name, parents=[self], id_params=self.id_params, output_dataset_id=coded_name, **params)

    def reason(self, task: str) -> LogicalDataset:
        """
        Apply a reasoning operation to the dataset based on the given task.
        """
        reasoned_name = f"ReasonOperation{self.id_params['reason_id'] + 1}"
        self.id_params["reason_id"] += 1
        params = {"operator": "Reason", "description": f"Reasoned {self.name} for task: {task}", "task": task}
        return LogicalDataset(reasoned_name, parents=[self], id_params=self.id_params, output_dataset_id=reasoned_name, **params)

    def sem_aggregate(self, task: str, agg_fields: list[dict]) -> LogicalDataset:
        """
        Apply a semantic aggregation to the dataset based on the given aggregation fields.
        """
        agg_name = f"AggregateOperation{self.id_params['sem_agg_id'] + 1}"
        self.id_params["sem_agg_id"] += 1
        for field_dict in agg_fields:
            field_dict["type"] = field_dict["type"].__name__
        params = {"operator": "SemanticAgg", "description": f"Aggregated {self.name} on fields: {agg_fields}", "task": task, "agg_fields": agg_fields}
        return LogicalDataset(agg_name, parents=[self], id_params=self.id_params, output_dataset_id=agg_name, **params)

    def sem_filter(self, condition: str) -> LogicalDataset:
        """
        Apply a semantic filter to the dataset based on the given condition.
        """
        filtered_name = f"FilterOperation{self.id_params['sem_filter_id'] + 1}"
        self.id_params["sem_filter_id"] += 1
        params = {"operator": "SemanticFilter", "description": f"Filtered {self.name} by condition: {condition}", "condition": condition}
        return LogicalDataset(filtered_name, parents=[self], id_params=self.id_params, output_dataset_id=filtered_name, **params)

    def sem_map(self, field: str, type: type, description: str) -> LogicalDataset:
        """
        Apply a semantic map to the dataset, adding or transforming a field.
        """
        mapped_name = f"MapOperation{self.id_params['sem_map_id'] + 1}"
        self.id_params["sem_map_id"] += 1
        params = {
            "operator": "SemanticMap",
            "description": f"Created field {field} with type {type.__name__} and description {description}",
            "field": field,
            "type": type.__name__,
            "field_desc": description,
        }
        return LogicalDataset(mapped_name, parents=[self], id_params=self.id_params, output_dataset_id=mapped_name, **params)

    def sem_flat_map(self, field: str, type: type, description: str) -> LogicalDataset:
        """
        Apply a semantic flat map to the dataset, expanding a field into multiple entries.
        """
        flat_mapped_name = f"FlatMapOperation{self.id_params['sem_flat_map_id'] + 1}"
        self.id_params["sem_flat_map_id"] += 1
        params = {
            "operator": "SemanticFlatMap",
            "description": f"Flat mapped field {field} with type {type.__name__} and description {description}",
            "field": field,
            "type": type.__name__,
            "field_desc": description,
        }
        return LogicalDataset(flat_mapped_name, parents=[self], id_params=self.id_params, output_dataset_id=flat_mapped_name, **params)

    def sem_groupby(self, gby_fields: list[dict], agg_fields: list[dict]) -> LogicalDataset:
        """
        Apply a semantic group by operation to the dataset.
        """
        gby_name = f"GroupByOperation{self.id_params['sem_groupby_id'] + 1}"
        self.id_params["sem_groupby_id"] += 1
        gby_field_names = [field['name'] for field in gby_fields]
        agg_field_names = [field['name'] for field in agg_fields]
        for field_dict in gby_fields:
            field_dict["type"] = field_dict["type"].__name__
        for field_dict in agg_fields:
            field_dict["type"] = field_dict["type"].__name__
        params = {
            "operator": "SemanticGroupBy",
            "description": f"Grouped {self.name} by fields {gby_field_names} with aggregations on {agg_field_names}",
            "gby_fields": gby_fields,
            "agg_fields": agg_fields,
        }
        return LogicalDataset(gby_name, parents=[self], id_params=self.id_params, output_dataset_id=gby_name, **params)

    def sem_join(self, other: LogicalDataset, condition: str) -> LogicalDataset:
        """
        Apply a semantic join with another dataset based on the given condition.
        """
        joined_name = f"JoinOperation{self.id_params['sem_join_id'] + 1}"
        self.id_params["sem_join_id"] += 1
        params = {
            "operator": "SemanticJoin",
            "description": f"Joined {self.name} with {other.name} on condition: {condition}",
            "condition": condition,
        }
        return LogicalDataset(joined_name, parents=[self, other], id_params=self.id_params, output_dataset_id=joined_name, **params)

    def sem_topk(self, search_str: str, k: int = 5) -> LogicalDataset:
        """
        Apply a semantic top-k operation with the given search string and k value.
        """
        top_k_name = f"TopKOperation{self.id_params['sem_topk_id'] + 1}"
        self.id_params["sem_topk_id"] += 1
        params = {
            "operator": "SemanticTopK",
            "description": f"Top-{k} items from {self.name} for search string: {search_str}",
            "search_str": search_str,
            "k": k,
        }
        return LogicalDataset(top_k_name, parents=[self], id_params=self.id_params, output_dataset_id=top_k_name, **params)
