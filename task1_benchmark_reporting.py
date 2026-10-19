"""Stable W&B cells for heterogeneous diagnostic records; raw JSON stays structured."""
import json


def table_rows(columns, rows):
    def cell(value):
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        return value
    return [[cell(row.get(column)) for column in columns] for row in rows]
