"""Delivery verification lifecycle tracker and latency analytics for Phase 4.

Separates MQTT protocol-level QoS confirmations from true application-level
receipt verifications. Calculates publication, subscriber receipt, and end-to-end latencies.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import pandas as pd

from rl_ack.common.enums import DeliveryStatus
from rl_ack.common.models import AckEvent, AlertMessage


@dataclass
class DeliveryRecord:
    """Detailed lifecycle tracking record for an individual alert."""

    alert_id: str
    incident_id: Optional[str]
    sensor_id: str
    severity: str
    status: DeliveryStatus = DeliveryStatus.GENERATED
    qos: int = 1
    topic: str = "rlack/kitchen/alerts"

    # Wall-clock timestamps in epoch seconds
    created_time: float = field(default_factory=time.time)
    publish_attempt_time: Optional[float] = None
    broker_ack_time: Optional[float] = None
    subscriber_receipt_time: Optional[float] = None
    delivery_confirmed_time: Optional[float] = None
    error_message: Optional[str] = None

    @property
    def publication_latency_ms(self) -> Optional[float]:
        """Latency from publish attempt to broker acknowledgement (PUBACK)."""
        if self.publish_attempt_time and self.broker_ack_time:
            return max(0.0, (self.broker_ack_time - self.publish_attempt_time) * 1000.0)
        return None

    @property
    def subscriber_receipt_latency_ms(self) -> Optional[float]:
        """Latency from publish attempt to subscriber receiving the message."""
        if self.publish_attempt_time and self.subscriber_receipt_time:
            return max(0.0, (self.subscriber_receipt_time - self.publish_attempt_time) * 1000.0)
        return None

    @property
    def app_ack_latency_ms(self) -> Optional[float]:
        """Latency from subscriber receipt to publisher receiving application-level ACK."""
        if self.subscriber_receipt_time and self.delivery_confirmed_time:
            return max(0.0, (self.delivery_confirmed_time - self.subscriber_receipt_time) * 1000.0)
        return None

    @property
    def end_to_end_latency_ms(self) -> Optional[float]:
        """Round-trip delivery latency: publish initiation to application confirmation."""
        if self.publish_attempt_time and self.delivery_confirmed_time:
            return max(0.0, (self.delivery_confirmed_time - self.publish_attempt_time) * 1000.0)
        elif self.publish_attempt_time and self.subscriber_receipt_time:
            return max(0.0, (self.subscriber_receipt_time - self.publish_attempt_time) * 1000.0)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize delivery tracking record to dictionary."""
        return {
            "alert_id": self.alert_id,
            "incident_id": self.incident_id,
            "sensor_id": self.sensor_id,
            "severity": self.severity,
            "status": self.status.value,
            "qos": self.qos,
            "topic": self.topic,
            "created_time": self.created_time,
            "publish_attempt_time": self.publish_attempt_time,
            "broker_ack_time": self.broker_ack_time,
            "subscriber_receipt_time": self.subscriber_receipt_time,
            "delivery_confirmed_time": self.delivery_confirmed_time,
            "publication_latency_ms": self.publication_latency_ms,
            "subscriber_receipt_latency_ms": self.subscriber_receipt_latency_ms,
            "end_to_end_latency_ms": self.end_to_end_latency_ms,
            "error_message": self.error_message,
        }


class DeliveryTracker:
    """Stateful delivery manager tracking alert verification stages and measuring latencies."""

    def __init__(self, max_records: int = 500) -> None:
        self.max_records = max_records
        self.records: Dict[str, DeliveryRecord] = {}
        self.events: List[AckEvent] = []

    def record_created(self, alert: AlertMessage) -> DeliveryRecord:
        """Register a freshly generated alert into the delivery tracking state."""
        rec = DeliveryRecord(
            alert_id=alert.alert_id,
            incident_id=alert.incident_id,
            sensor_id=alert.sensor_id,
            severity=alert.severity.value,
            status=DeliveryStatus.GENERATED,
            created_time=time.time(),
        )
        self.records[alert.alert_id] = rec
        self._record_event(alert.alert_id, "alert_created", time.time())
        return rec

    def record_publish_attempt(
        self,
        alert_id: str,
        qos: int,
        topic: str,
        timestamp: Optional[float] = None,
    ) -> Optional[DeliveryRecord]:
        """Record initiation of MQTT publication."""
        t = time.time() if timestamp is None else timestamp
        rec = self.records.get(alert_id)
        if rec is None:
            rec = DeliveryRecord(
                alert_id=alert_id,
                incident_id=None,
                sensor_id="UNKNOWN",
                severity="unknown",
            )
            self.records[alert_id] = rec

        rec.status = DeliveryStatus.PUBLISH_PENDING
        rec.qos = qos
        rec.topic = topic
        rec.publish_attempt_time = t
        self._record_event(alert_id, "publish_attempted", t, details={"qos": qos, "topic": topic})
        return rec

    def record_broker_ack(
        self, alert_id: str, timestamp: Optional[float] = None
    ) -> Optional[DeliveryRecord]:
        """Record broker protocol-level acknowledgement (PUBACK / PUBCOMP)."""
        t = time.time() if timestamp is None else timestamp
        rec = self.records.get(alert_id)
        if rec is not None:
            rec.broker_ack_time = t
            rec.status = DeliveryStatus.PUBLISHED
            latency = rec.publication_latency_ms
            self._record_event(
                alert_id,
                "broker_acknowledged",
                t,
                latency_ms=latency,
            )
        return rec

    def record_subscriber_receipt(
        self, alert_id: str, timestamp: Optional[float] = None
    ) -> Optional[DeliveryRecord]:
        """Record subscriber client successfully receiving and parsing the alert."""
        t = time.time() if timestamp is None else timestamp
        rec = self.records.get(alert_id)
        if rec is None:
            # If subscriber received alert before tracker saw local creation
            rec = DeliveryRecord(
                alert_id=alert_id,
                incident_id=None,
                sensor_id="UNKNOWN",
                severity="unknown",
            )
            self.records[alert_id] = rec

        rec.subscriber_receipt_time = t
        rec.status = DeliveryStatus.RECEIVED
        latency = rec.subscriber_receipt_latency_ms
        self._record_event(
            alert_id,
            "subscriber_received",
            t,
            latency_ms=latency,
        )
        return rec

    def record_delivery_confirmed(
        self, alert_id: str, timestamp: Optional[float] = None
    ) -> Optional[DeliveryRecord]:
        """Record confirmation of application-level receipt acknowledgement."""
        t = time.time() if timestamp is None else timestamp
        rec = self.records.get(alert_id)
        if rec is not None:
            rec.delivery_confirmed_time = t
            rec.status = DeliveryStatus.DELIVERY_CONFIRMED
            latency = rec.end_to_end_latency_ms
            self._record_event(
                alert_id,
                "delivery_confirmed",
                t,
                latency_ms=latency,
            )
        return rec

    def record_failure(
        self, alert_id: str, error_message: str, timestamp: Optional[float] = None
    ) -> Optional[DeliveryRecord]:
        """Record delivery failure."""
        t = time.time() if timestamp is None else timestamp
        rec = self.records.get(alert_id)
        if rec is not None:
            rec.status = DeliveryStatus.FAILED
            rec.error_message = error_message
            self._record_event(
                alert_id,
                "delivery_failed",
                t,
                success=False,
                details={"error": error_message},
            )
        return rec

    def check_timeouts(self, timeout_s: float = 5.0) -> List[str]:
        """Check for alerts that have been in PUBLISH_PENDING or PUBLISHED beyond timeout_s."""
        now = time.time()
        timed_out_ids: List[str] = []
        for alert_id, rec in self.records.items():
            if rec.status in (
                DeliveryStatus.PUBLISH_PENDING,
                DeliveryStatus.PUBLISHED,
                DeliveryStatus.RECEIVED,
            ):
                ref_time = rec.publish_attempt_time or rec.created_time
                if (now - ref_time) > timeout_s:
                    rec.status = DeliveryStatus.TIMED_OUT
                    rec.error_message = f"Delivery receipt timed out after {timeout_s:.1f}s"
                    timed_out_ids.append(alert_id)
                    self._record_event(
                        alert_id,
                        "delivery_timed_out",
                        now,
                        success=False,
                        details={"timeout_s": timeout_s},
                    )
        return timed_out_ids

    def _record_event(
        self,
        alert_id: str,
        event_type: str,
        timestamp: float,
        latency_ms: Optional[float] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Internal helper to log structured AckEvent."""
        event_id = f"EVT-{len(self.events) + 1:05d}"
        evt = AckEvent(
            event_id=event_id,
            alert_id=alert_id,
            event_type=event_type,
            timestamp=timestamp,
            latency_ms=latency_ms,
            success=success,
            details=details or {},
        )
        self.events.append(evt)
        if len(self.events) > self.max_records * 4:
            self.events = self.events[-self.max_records * 2 :]

    def get_metrics(self) -> Dict[str, Any]:
        """Calculate system-wide aggregate delivery KPIs."""
        total = len(self.records)
        published = sum(
            1
            for r in self.records.values()
            if r.publish_attempt_time is not None
        )
        broker_acked = sum(
            1 for r in self.records.values() if r.broker_ack_time is not None
        )
        received = sum(
            1
            for r in self.records.values()
            if r.subscriber_receipt_time is not None
        )
        confirmed = sum(
            1
            for r in self.records.values()
            if r.status
            in (DeliveryStatus.DELIVERY_CONFIRMED, DeliveryStatus.DELIVERED)
        )
        pending = sum(
            1
            for r in self.records.values()
            if r.status
            in (
                DeliveryStatus.PUBLISH_PENDING,
                DeliveryStatus.PUBLISHED,
                DeliveryStatus.RECEIVED,
            )
        )
        timed_out = sum(
            1
            for r in self.records.values()
            if r.status in (DeliveryStatus.TIMED_OUT, DeliveryStatus.TIMEOUT)
        )
        failed = sum(
            1 for r in self.records.values() if r.status == DeliveryStatus.FAILED
        )

        latencies = [
            r.end_to_end_latency_ms
            for r in self.records.values()
            if r.end_to_end_latency_ms is not None
        ]
        avg_latency = sum(latencies) / len(latencies) if latencies else None

        return {
            "total_alerts": total,
            "total_published": published,
            "broker_acknowledged": broker_acked,
            "subscriber_received": received,
            "confirmed_deliveries": confirmed,
            "pending_receipts": pending,
            "timed_out": timed_out,
            "failed": failed,
            "avg_e2e_latency_ms": avg_latency,
        }

    def to_dataframe(self, limit: int = 50) -> pd.DataFrame:
        """Format recent delivery records for Streamlit display."""
        if not self.records:
            return pd.DataFrame(
                columns=[
                    "Alert ID",
                    "Sensor ID",
                    "Severity",
                    "Status",
                    "QoS",
                    "Publish Latency",
                    "Receipt Latency",
                    "E2E Latency",
                    "Error",
                ]
            )

        rows = []
        for r in list(self.records.values())[-limit:]:
            pub_lat_str = (
                f"{r.publication_latency_ms:.1f} ms"
                if r.publication_latency_ms is not None
                else "—"
            )
            rec_lat_str = (
                f"{r.subscriber_receipt_latency_ms:.1f} ms"
                if r.subscriber_receipt_latency_ms is not None
                else "—"
            )
            e2e_lat_str = (
                f"{r.end_to_end_latency_ms:.1f} ms"
                if r.end_to_end_latency_ms is not None
                else "—"
            )
            rows.append(
                {
                    "Alert ID": r.alert_id,
                    "Sensor ID": r.sensor_id,
                    "Severity": r.severity.upper(),
                    "Status": r.status.value.upper(),
                    "QoS": r.qos,
                    "Publish Latency": pub_lat_str,
                    "Receipt Latency": rec_lat_str,
                    "E2E Latency": e2e_lat_str,
                    "Error": r.error_message or "None",
                }
            )

        return pd.DataFrame(rows)

    def reset(self) -> None:
        """Clear all delivery tracking records and event logs."""
        self.records.clear()
        self.events.clear()
