"""Citable identifiers and release attribution.

Sites contribute data, compute and review effort; if that work is not citable,
it is not counted by the systems academic medicine actually rewards. This module
mints stable, content-derived identifiers for releases and records who did what.

HONESTY NOTE: the identifiers minted here are DOI-*style* -- they have the same
``prefix/suffix`` shape and the same role in a citation -- but they are minted
locally and are **not registered DOIs**. Registering real DOIs requires
membership of a registration agency (DataCite or Crossref) and a deposit step
that this package does not perform. :attr:`CitableIdentifier.registered` says
which kind you are holding; :meth:`CitableIdentifier.for_registered_doi` is the
constructor to use once a real DOI has been deposited elsewhere.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trustfed.incentives.errors import IdentifierError

#: Default local (unregistered) namespace for minted identifiers.
LOCAL_PREFIX = "openmed.local"


@dataclass(frozen=True)
class CitableIdentifier:
    """A ``prefix/suffix`` identifier for a release.

    Attributes
    ----------
    registered:
        False for locally minted identifiers (the default), True only when the
        caller is recording a DOI that was genuinely deposited with a
        registration agency.
    """

    prefix: str
    suffix: str
    registered: bool = False

    def __post_init__(self) -> None:
        if not self.prefix or "/" in self.prefix:
            raise IdentifierError("prefix must be non-empty and contain no '/'")
        if not self.suffix or "/" in self.suffix:
            raise IdentifierError("suffix must be non-empty and contain no '/'")

    @property
    def value(self) -> str:
        """Return the identifier as ``prefix/suffix``."""
        return f"{self.prefix}/{self.suffix}"

    @property
    def uri(self) -> str:
        """Return a URI form: ``doi:`` when registered, ``oid:`` when local."""
        return ("doi:" if self.registered else "oid:") + self.value

    @classmethod
    def mint(
        cls, content_id: str, *, prefix: str = LOCAL_PREFIX, length: int = 10
    ) -> "CitableIdentifier":
        """Derive a deterministic local identifier from a content id.

        The same bundle id always mints the same identifier, so two sites that
        publish the same release cite it identically.
        """
        if not content_id:
            raise IdentifierError("content_id must not be empty")
        digest = hashlib.sha256(content_id.encode("utf-8")).hexdigest()[:length]
        return cls(prefix=prefix, suffix=digest, registered=False)

    @classmethod
    def for_registered_doi(cls, prefix: str, suffix: str) -> "CitableIdentifier":
        """Record a DOI that was actually registered with an agency elsewhere."""
        return cls(prefix=prefix, suffix=suffix, registered=True)

    @classmethod
    def parse(cls, value: str, *, registered: bool = False) -> "CitableIdentifier":
        """Parse ``prefix/suffix`` (optionally ``doi:``/``oid:``-prefixed)."""
        text = value.split(":", 1)[1] if value.startswith(("doi:", "oid:")) else value
        if text.count("/") != 1:
            raise IdentifierError(f"malformed identifier: {value!r}")
        prefix, suffix = text.split("/")
        return cls(
            prefix=prefix,
            suffix=suffix,
            registered=registered or value.startswith("doi:"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "prefix": self.prefix,
            "suffix": self.suffix,
            "registered": self.registered,
            "uri": self.uri,
        }


@dataclass(frozen=True)
class Attribution:
    """One contributor's role in a release.

    ``role`` is free text so deployments can adopt CRediT-style vocabularies;
    ``share`` is an optional relative weight used only for display.
    """

    actor: str
    role: str
    institution: str = ""
    share: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "actor": self.actor,
            "role": self.role,
            "institution": self.institution,
            "share": self.share,
        }


@dataclass(frozen=True)
class ReleaseAttribution:
    """Citable record of who produced a release.

    Contains only what the caller supplied: this module never invents authors,
    affiliations, or publication venues.
    """

    bundle_id: str
    title: str
    year: int
    identifier: CitableIdentifier
    contributors: Tuple[Attribution, ...] = ()
    version: str = ""

    @classmethod
    def for_bundle(
        cls,
        bundle_id: str,
        *,
        title: str,
        year: int,
        contributors: Sequence[Attribution] = (),
        version: str = "",
        prefix: str = LOCAL_PREFIX,
    ) -> "ReleaseAttribution":
        """Mint a local identifier for ``bundle_id`` and attach attribution."""
        return cls(
            bundle_id=bundle_id,
            title=title,
            year=year,
            identifier=CitableIdentifier.mint(bundle_id, prefix=prefix),
            contributors=tuple(contributors),
            version=version,
        )

    def actors(self) -> Tuple[str, ...]:
        """Return the distinct contributor ids, in listed order."""
        seen: list = []
        for contributor in self.contributors:
            if contributor.actor not in seen:
                seen.append(contributor.actor)
        return tuple(seen)

    def to_citation(self) -> str:
        """Render a citation string for this release.

        Locally minted identifiers are rendered with the ``oid:`` scheme and an
        explicit note, so a reader is never misled into thinking a DOI was
        registered.
        """
        authors = ", ".join(self.actors()) if self.contributors else "(no contributors listed)"
        version = f" (version {self.version})" if self.version else ""
        note = "" if self.identifier.registered else " [locally minted, unregistered]"
        return (
            f"{authors} ({self.year}). {self.title}{version}. "
            f"{self.identifier.uri}{note}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "bundle_id": self.bundle_id,
            "title": self.title,
            "year": self.year,
            "version": self.version,
            "identifier": self.identifier.to_dict(),
            "contributors": [c.to_dict() for c in self.contributors],
            "citation": self.to_citation(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReleaseAttribution":
        """Rebuild from :meth:`to_dict` output."""
        ident = data["identifier"]
        return cls(
            bundle_id=str(data["bundle_id"]),
            title=str(data["title"]),
            year=int(data["year"]),
            identifier=CitableIdentifier(
                prefix=str(ident["prefix"]),
                suffix=str(ident["suffix"]),
                registered=bool(ident.get("registered", False)),
            ),
            contributors=tuple(
                Attribution(
                    actor=str(c["actor"]),
                    role=str(c["role"]),
                    institution=str(c.get("institution", "")),
                    share=float(c.get("share", 0.0)),
                )
                for c in data.get("contributors", ())
            ),
            version=str(data.get("version", "")),
        )


__all__ = [
    "LOCAL_PREFIX",
    "Attribution",
    "CitableIdentifier",
    "ReleaseAttribution",
]
