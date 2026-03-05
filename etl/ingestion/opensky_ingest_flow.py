import io
import json
import os
import datetime as dt
from typing import Any, Dict, List, Optional

import requests
from google.cloud import bigquery, storage
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
def upload_to_gcs_bronze(
    bucket_name: str,
    rows: List[Dict[str, Any]],
    snapshot_time: dt.datetime,
) -> None:
    """Upload the raw snapshot as NDJSON to GCS bronze, partitioned by date/hour."""
    logger = get_run_logger()
    if not rows:
        logger.info("No rows to upload.")
        return

    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode("utf-8")
    file_obj = io.BytesIO(payload)

    year = snapshot_time.strftime("%Y")
    month = snapshot_time.strftime("%m")
    day = snapshot_time.strftime("%d")
    hour = snapshot_time.strftime("%H")
    timestamp = snapshot_time.strftime("%Y%m%d_%H%M%S")
    blob_name = f"bronze/ingest_year={year}/ingest_month={month}/ingest_day={day}/ingest_hour={hour}/snapshot_{timestamp}.ndjson"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(file_obj, content_type="application/x-ndjson")
    logger.info("Uploaded %d rows to gs://%s/%s", len(rows), bucket_name, blob_name)


@task(retries=3, retry_delay_seconds=10)
def load_bigquery(project_id: str, dataset_id: str, table_name: str, rows: list[dict]) -> None:
    logger = get_run_logger()
    if not rows:
        logger.info("No rows to load.")
        return

    table_id = f"{project_id}.{dataset_id}.{table_name}"
    client = bigquery.Client(project=project_id)

    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode("utf-8")
    file_obj = io.BytesIO(payload)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        autodetect=False,
    )

    logger.info("Loading %d rows into %s", len(rows), table_id)
    job = client.load_table_from_file(file_obj, table_id, job_config=job_config)
    job.result()
    logger.info("Load job finished: %s", job.job_id)


@flow(name="opensky_ingest_flow")
def opensky_ingest_flow():
    gcs_bucket = env_str("GCS_BUCKET")

    opensky_url = env_str("OPENSKY_URL", "https://opensky-network.org/api/states/all")

    params = {
        "lamin": env_float("BBOX_LAMIN", 41.0),
        "lamax": env_float("BBOX_LAMAX", 51.5),
        "lomin": env_float("BBOX_LOMIN", -5.5),
        "lomax": env_float("BBOX_LOMAX", 9.5),
    }

    snapshot = fetch_opensky_states(opensky_url, params)
    rows = transform_snapshot(snapshot)

    if rows:
        snapshot_time = dt.datetime.fromisoformat(rows[0]["ingest_ts"])
    else:
        snapshot_time = dt.datetime.utcnow()

    upload_to_gcs_bronze(gcs_bucket, rows, snapshot_time)



if __name__ == "__main__":
    opensky_ingest_flow()