"""
providers/base.py - Provider protocol és lifecycle alapok.

Minden provider implementálja a BaseProvider protokollt.
Ez a réteg biztosítja:
  - Egységes lifecycle: load → run → release
  - VRAM-tudatos szekvenciális futtatás
  - Confidence-gated opcionális futtatás
  - Struktúrált ProviderResult
  - Hot-swap támogatás (GUI future)
  - Thread-safe tervezés (future parallel)

Tervezési elvek:
  - Nincs globális mutable state a providerekben
  - Minden provider önálló lifecycle-lal rendelkezik
  - A pipeline csak a ProviderResult-ot látja, nem a belső implementációt
  - Failure isolation: egy provider hibája nem törli a pipeline-t
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Generic, TypeVar, Generator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER STÁTUSZ
# ══════════════════════════════════════════════════════════════════════════════

class ProviderStatus(Enum):
    UNLOADED   = auto()   # modell nincs betöltve
    LOADING    = auto()   # betöltés folyamatban
    READY      = auto()   # kész, futtatható
    RUNNING    = auto()   # éppen fut
    ERROR      = auto()   # hiba állapot
    RELEASED   = auto()   # VRAM felszabadítva


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER RESULT – struktúrált output
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProviderResult(Generic[T]):
    """
    Egységes provider output struktúra.

    Minden provider ProviderResult-ot ad vissza –
    a pipeline nem látja a belső implementációt.

    Fields:
        success:      sikeres-e a futtatás
        data:         a tényleges output (típus provider-specifikus)
        confidence:   0.0..1.0 – output megbízhatósága
        provider_id:  melyik provider generálta
        elapsed_ms:   futási idő milliszekundumban
        error:        hiba üzenet ha success=False
        metadata:     extra mezők (provider-specifikus debug info)
    """
    success:     bool
    data:        Optional[T]        = None
    confidence:  float              = 0.0
    provider_id: str                = "unknown"
    elapsed_ms:  float              = 0.0
    error:       Optional[str]      = None
    metadata:    dict[str, Any]     = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        data: T,
        confidence: float = 1.0,
        provider_id: str = "unknown",
        elapsed_ms: float = 0.0,
        **meta,
    ) -> "ProviderResult[T]":
        """Sikeres eredmény factory."""
        return cls(
            success=True,
            data=data,
            confidence=confidence,
            provider_id=provider_id,
            elapsed_ms=elapsed_ms,
            metadata=meta,
        )

    @classmethod
    def fail(
        cls,
        error: str,
        provider_id: str = "unknown",
        elapsed_ms: float = 0.0,
    ) -> "ProviderResult[T]":
        """Hibás eredmény factory."""
        return cls(
            success=False,
            data=None,
            confidence=0.0,
            provider_id=provider_id,
            elapsed_ms=elapsed_ms,
            error=error,
        )

    @classmethod
    def skip(
        cls,
        reason: str,
        provider_id: str = "unknown",
    ) -> "ProviderResult[T]":
        """
        Kihagyott futtatás (confidence-gate vagy disabled).
        Nem hiba – a pipeline folytatódik.
        """
        return cls(
            success=True,
            data=None,
            confidence=0.0,
            provider_id=provider_id,
            error=None,
            metadata={"skipped": True, "reason": reason},
        )

    @property
    def skipped(self) -> bool:
        return self.metadata.get("skipped", False)


# ══════════════════════════════════════════════════════════════════════════════
# BASE PROVIDER – ABC
# ══════════════════════════════════════════════════════════════════════════════

class BaseProvider(ABC):
    """
    Minden provider ősosztálya.

    Lifecycle:
      1. __init__()   – konfiguráció, NEM tölt be modellt
      2. load()       – modell betöltés VRAM-ba
      3. run(...)     – inference futtatás
      4. release()    – VRAM felszabadítás

    Szekvenciális VRAM management:
      Az orchestrator hívja: load → run → release
      Soha nem marad két nagy modell egyszerre VRAM-ban.

    Confidence gate:
      Ha a bemenet confidence >= min_input_confidence: fut
      Különben: ProviderResult.skip() visszaadva

    GUI hot-swap:
      A provider bármikor lecserélhető futás közben
      (release régi → load új → folytatás)
    """

    def __init__(
        self,
        provider_id: str,
        min_input_confidence: float = 0.0,
        enabled: bool = True,
    ) -> None:
        self.provider_id          = provider_id
        self.min_input_confidence = min_input_confidence
        self.enabled              = enabled
        self._status              = ProviderStatus.UNLOADED
        self._load_time_ms        = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """
        Modell betöltés VRAM-ba.

        Returns:
            True ha sikeres, False ha hiba.
        """
        if self._status == ProviderStatus.READY:
            return True
        if not self.enabled:
            logger.debug(f"[{self.provider_id}] disabled – load kihagyva")
            return False

        self._status = ProviderStatus.LOADING
        t0 = time.perf_counter()
        try:
            success = self._load_model()
            elapsed = (time.perf_counter() - t0) * 1000
            self._load_time_ms = elapsed
            if success:
                self._status = ProviderStatus.READY
                logger.info(
                    f"[{self.provider_id}] betöltve "
                    f"({elapsed:.0f}ms)"
                )
            else:
                self._status = ProviderStatus.ERROR
                logger.warning(f"[{self.provider_id}] betöltés sikertelen")
            return success
        except Exception as e:
            self._status = ProviderStatus.ERROR
            logger.error(f"[{self.provider_id}] load hiba: {e}")
            return False

    def release(self) -> None:
        """
        VRAM felszabadítás.

        Biztonságos ha nem volt betöltve.
        """
        if self._status not in (ProviderStatus.READY, ProviderStatus.ERROR):
            return
        try:
            self._release_model()
            self._status = ProviderStatus.RELEASED
            logger.debug(f"[{self.provider_id}] VRAM felszabadítva")
        except Exception as e:
            logger.warning(f"[{self.provider_id}] release hiba: {e}")
        finally:
            self._cleanup_memory()

    @property
    def is_ready(self) -> bool:
        return self._status == ProviderStatus.READY

    @property
    def status(self) -> ProviderStatus:
        return self._status

    # ── Confidence gate ───────────────────────────────────────────────────────

    def should_run(self, input_confidence: float = 1.0) -> bool:
        """
        Futtatható-e ez a provider a bemenet confidence alapján.

        Confidence-gated: ha az input confidence alacsony
        és a provider "remediation" szerepű (pl. Gemma OCR correction),
        akkor fut. Ha magas confidence → skip.
        """
        if not self.enabled:
            return False
        if self._status != ProviderStatus.READY:
            return False
        return True

    # ── Timing wrapper ────────────────────────────────────────────────────────

    def timed_run(self, *args, **kwargs) -> ProviderResult:
        """
        run() hívás automatikus időmérésssel.

        A subclass a run()-t implementálja,
        a timed_run() hívható az orchestratorból.
        """
        if not self.is_ready:
            return ProviderResult.fail(
                f"Provider nem ready: {self._status.name}",
                provider_id=self.provider_id,
            )

        self._status = ProviderStatus.RUNNING
        t0 = time.perf_counter()
        try:
            result = self.run(*args, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            result.elapsed_ms  = elapsed
            result.provider_id = self.provider_id
            self._status = ProviderStatus.READY
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            self._status = ProviderStatus.READY
            logger.warning(f"[{self.provider_id}] run hiba: {e}")
            return ProviderResult.fail(
                str(e),
                provider_id=self.provider_id,
                elapsed_ms=elapsed,
            )

    # ── Abstract metódusok ────────────────────────────────────────────────────

    @abstractmethod
    def _load_model(self) -> bool:
        """Modell betöltés implementálása. True = sikeres."""
        ...

    @abstractmethod
    def _release_model(self) -> None:
        """Modell VRAM felszabadítás implementálása."""
        ...

    @abstractmethod
    def run(self, *args, **kwargs) -> ProviderResult:
        """
        Inference futtatás.

        A subclass implementálja – a timed_run() hívja.
        """
        ...

    # ── Memory cleanup ────────────────────────────────────────────────────────

    @staticmethod
    def _cleanup_memory() -> None:
        """GPU + CPU memory cleanup."""
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# LIGHTWEIGHT PROVIDER – modell nélküli providerekhez
# ══════════════════════════════════════════════════════════════════════════════

class LightweightProvider(BaseProvider):
    """
    Modell nélküli provider (API hívás, rule-based, stb.)

    Nem igényel VRAM – load/release trivialis.
    Példák: OllamaProvider (API call), DeepLProvider (HTTP), rule-based cleaner.
    """

    def _load_model(self) -> bool:
        """Nincs modell – azonnal ready."""
        return True

    def _release_model(self) -> None:
        """Nincs mit felszabadítani."""
        pass


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER REGISTRY – hot-swap és discovery
# ══════════════════════════════════════════════════════════════════════════════

class ProviderRegistry:
    """
    Provider regisztráció és discovery.

    Feladatai:
      - Elérhető providerek nyilvántartása
      - Hot-swap: futás közben lecserélhető
      - GUI integrációhoz: listázható, váltható
      - Orchestrator nem hardcode-olja a provider neveket

    Thread-safe: csak olvasás párhuzamosan, írás sequentiálisan.
    """

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        """Provider regisztrálása."""
        self._providers[provider.provider_id] = provider
        logger.debug(f"Provider regisztrálva: {provider.provider_id}")

    def get(self, provider_id: str) -> Optional[BaseProvider]:
        """Provider lekérés ID alapján."""
        return self._providers.get(provider_id)

    def get_ready(self, provider_id: str) -> Optional[BaseProvider]:
        """Csak ready státuszú provider."""
        p = self._providers.get(provider_id)
        return p if (p and p.is_ready) else None

    def swap(
        self,
        old_id: str,
        new_provider: BaseProvider,
    ) -> None:
        """
        Provider hot-swap.

        1. Régi provider release-elve
        2. Új provider regisztrálva
        """
        old = self._providers.get(old_id)
        if old:
            old.release()
        self._providers[new_provider.provider_id] = new_provider
        logger.info(
            f"Provider swap: {old_id} → {new_provider.provider_id}")

    def release_all(self) -> None:
        """Minden provider VRAM felszabadítása."""
        for p in self._providers.values():
            p.release()
        logger.info("Minden provider felszabadítva")

    def status_summary(self) -> dict[str, str]:
        """Állapot összefoglaló (logoláshoz, GUI-hoz)."""
        return {
            pid: p.status.name
            for pid, p in self._providers.items()
        }

    def list_ids(self) -> list[str]:
        return list(self._providers.keys())


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE GATE HELPER
# ══════════════════════════════════════════════════════════════════════════════

class ConfidenceGate:
    """
    Confidence-alapú provider futtatási döntés.

    Példa:
      gate = ConfidenceGate(threshold=0.70)
      if gate.should_remediate(ocr_confidence):
          result = gemma_provider.timed_run(text, image)

    A pipeline determinisztikus marad:
    magas confidence → Gemma skip
    alacsony confidence → Gemma fut
    """

    def __init__(
        self,
        threshold: float,
        mode: str = "below",  # "below" = fut ha confidence < threshold
    ) -> None:
        self.threshold = threshold
        self.mode      = mode

    def should_run(self, confidence: float) -> bool:
        """
        Futtatható-e a remediation provider.

        Args:
            confidence: az előző stage output confidence-ja

        Returns:
            True ha a provider futtatása indokolt.
        """
        if self.mode == "below":
            return confidence < self.threshold
        elif self.mode == "above":
            return confidence >= self.threshold
        return True

    def __repr__(self) -> str:
        return (
            f"ConfidenceGate(threshold={self.threshold}, "
            f"mode='{self.mode}')"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER CONTEXT – orchestrator lifecycle helper
# ══════════════════════════════════════════════════════════════════════════════

class ProviderContext:
    """
    Context manager a provider lifecycle-hoz.

    Használat:
        with ProviderContext(provider) as p:
            result = p.timed_run(...)
        # itt automatikusan release + VRAM cleanup

    Az orchestratorban biztonságos és tömör:
    nincs elfelejtett release() hívás.
    """

    def __init__(self, provider: BaseProvider) -> None:
        self._provider = provider

    def __enter__(self) -> BaseProvider:
        success = self._provider.load()
        if not success:
            raise RuntimeError(
                f"Provider betöltés sikertelen: "
                f"{self._provider.provider_id}"
            )
        return self._provider

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._provider.release()
        # False = nem nyeljük el a kivételt
        return False
