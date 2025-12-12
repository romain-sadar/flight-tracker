import io
import json
import os
import datetime as dt
from typing import Any, Dict, List, Optional

import requests
from google.cloud import bigquery
from prefect import flow, task
from prefect.logging import get_run_logger


def env_str(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v is not None else default


@task(retries=3, retry_delay_seconds=15)
def fetch_opensky_states(opensky_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info("Fetching OpenSky: %s params=%s", opensky_url, params)

    r = requests.get(opensky_url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def to_iso(dt_obj):
    return dt_obj.isoformat(timespec="seconds") if dt_obj else None

@task
def transform_snapshot(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    logger = get_run_logger()
    ingest_ts_dt = dt.datetime.utcfromtimestamp(snapshot["time"])

    rows: List[Dict[str, Any]] = []
    states = snapshot.get("states") or []

    for state in states:
        state = state + [None] * (17 - len(state))

        (
            icao24,
            callsign,
            origin_country,
            time_position,
            last_contact,
            lon,
            lat,
            baro_altitude,
            on_ground,
            velocity,
            heading,
            vertical_rate,
            *rest,
        ) = state

        tp_dt = dt.datetime.utcfromtimestamp(time_position) if time_position else None
        lc_dt = dt.datetime.utcfromtimestamp(last_contact) if last_contact else None

        rows.append(
            {
                "ingest_ts": to_iso(ingest_ts_dt),
                "icao24": icao24,
                "callsign": (callsign or "").strip(),
                "origin_country": origin_country,
                "time_position": to_iso(tp_dt),
                "last_contact": to_iso(lc_dt),
                "lon": lon,
                "lat": lat,
                "baro_altitude": baro_altitude,
                "on_ground": on_ground,
                "velocity": velocity,
                "heading": heading,
                "vertical_rate": vertical_rate,
            }
        )

    logger.info("Transformed rows: %d", len(rows))
    return rows


@task(retries=3, retry_delay_seconds=10)
def load_bigquery(project_id: str, dataset_id: str, table_name: str, rows: list[dict]) -> None:
    logger = get_run_logger()
    if not rows:
        logger.info("No rows to load.")
        return

    table_id = f"{project_id}.{dataset_id}.{table_name}"
    client = bigquery.Client(project=project_id)

    # NDJSON en mémoire (1 JSON par ligne)
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode("utf-8")
    file_obj = io.BytesIO(payload)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        autodetect=False,  # on a déjà le schéma dans la table
    )

    logger.info("Loading %d rows into %s via load job (NDJSON).", len(rows), table_id)
    job = client.load_table_from_file(
        file_obj,
        table_id,
        job_config=job_config,
    )
    job.result()  # attend la fin du job

    logger.info("Load job finished: %s", job.job_id)

@flow(name="opensky_ingest_flow")
def opensky_ingest_flow():
    project_id = env_str("GCP_PROJECT_ID")
    dataset_id = env_str("BQ_DATASET_ID")
    table_name = env_str("BQ_TABLE_NAME", "raw_flight_states")

    opensky_url = env_str("OPENSKY_URL", "https://opensky-network.org/api/states/all")

    params = {
        "lamin": env_float("BBOX_LAMIN", 41.0),
        "lamax": env_float("BBOX_LAMAX", 51.5),
        "lomin": env_float("BBOX_LOMIN", -5.5),
        "lomax": env_float("BBOX_LOMAX", 9.5),
    }

    snapshot = fetch_opensky_states(opensky_url, params)
    rows = transform_snapshot(snapshot)
    load_bigquery(project_id, dataset_id, table_name, rows)


if __name__ == "__main__":
    opensky_ingest_flow()
