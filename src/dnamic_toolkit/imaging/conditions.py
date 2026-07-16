"""Boolean ROI occupancy conditions and conditional binomial statistics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from dnamic_toolkit.imaging.binomial import estimate_probability_from_counts

OccupancyByImage = tuple[np.ndarray, ...]
ClauseTerm = tuple[int, int, int]


@dataclass(frozen=True)
class Always:
    """Condition that always evaluates to ``True``."""


@dataclass(frozen=True)
class Occupied:
    """One occupied trap at ``(image, roi)`` within one logical group."""

    image_index: int
    roi_index: int


@dataclass(frozen=True)
class Empty:
    """One empty trap at ``(image, roi)`` within one logical group."""

    image_index: int
    roi_index: int


@dataclass(frozen=True)
class Not:
    condition: object


@dataclass(frozen=True)
class And:
    conditions: tuple[object, ...]


@dataclass(frozen=True)
class Or:
    conditions: tuple[object, ...]


def all_of(*conditions) -> And:
    """Return the conjunction of ``conditions``."""

    return And(tuple(conditions))


def any_of(*conditions) -> Or:
    """Return the disjunction of ``conditions``."""

    return Or(tuple(conditions))


def _term_to_condition(image_index: int, roi_index: int, state: int):
    if int(state) == 1:
        return Occupied(int(image_index), int(roi_index))
    return Empty(int(image_index), int(roi_index))


def condition_from_clauses(
    clauses: list[list[ClauseTerm]] | tuple[tuple[ClauseTerm, ...], ...],
):
    """Return a condition from sparse DNF clauses.

    ``clauses`` is interpreted as outer OR, inner AND, with each term encoded
    as ``(image_index, roi_index, state)`` where state is 0 or 1.
    """

    if not clauses:
        return Always()

    disjuncts = []
    for clause in clauses:
        if not clause:
            disjuncts.append(Always())
            continue
        terms = tuple(_term_to_condition(*term) for term in clause)
        disjuncts.append(terms[0] if len(terms) == 1 else And(terms))

    return disjuncts[0] if len(disjuncts) == 1 else Or(tuple(disjuncts))


_TOKEN_RE = re.compile(
    r"""
    \s*
    (?:
        (?P<LPAREN>\()
        | (?P<RPAREN>\))
        | (?P<AND>&)
        | (?P<OR>\|)
        | (?P<NOT>!)
        | (?P<ATOM>(?:i)?\d+\[[^\[\]]*\])
    )
    """,
    re.VERBOSE,
)
_ATOM_RE = re.compile(r"(?P<image>(?:i)?\d+)\[(?P<terms>[^\[\]]*)\]$")
_ROI_TERM_RE = re.compile(r"(?P<neg>!)?(?P<roi>\d+)$")


@dataclass(frozen=True)
class _SyntaxToken:
    kind: str
    value: str
    position: int


def _tokenize_condition_syntax(text: str) -> list[_SyntaxToken]:
    tokens: list[_SyntaxToken] = []
    position = 0
    while position < len(text):
        match = _TOKEN_RE.match(text, position)
        if match is None:
            if text[position:].strip() == "":
                break
            raise ValueError(
                f"Invalid ROI condition syntax near {text[position:position + 16]!r}"
            )
        position = match.end()
        kind = match.lastgroup
        if kind is not None:
            tokens.append(_SyntaxToken(kind, match.group(kind), match.start(kind)))
    return tokens


class _ConditionSyntaxParser:
    def __init__(self, tokens: list[_SyntaxToken]):
        self._tokens = tokens
        self._index = 0

    def parse(self):
        if not self._tokens:
            return Always()
        condition = self._parse_or()
        if self._peek() is not None:
            token = self._peek()
            raise ValueError(
                f"Unexpected token {token.value!r} at position {token.position}"
            )
        return condition

    def _peek(self) -> _SyntaxToken | None:
        if self._index >= len(self._tokens):
            return None
        return self._tokens[self._index]

    def _accept(self, kind: str) -> _SyntaxToken | None:
        token = self._peek()
        if token is None or token.kind != kind:
            return None
        self._index += 1
        return token

    def _expect(self, kind: str) -> _SyntaxToken:
        token = self._accept(kind)
        if token is None:
            next_token = self._peek()
            if next_token is None:
                raise ValueError(f"Expected {kind} but reached end of expression")
            raise ValueError(
                f"Expected {kind} at position {next_token.position}, "
                f"got {next_token.value!r}"
            )
        return token

    def _parse_or(self):
        conditions = [self._parse_and()]
        while self._accept("OR") is not None:
            conditions.append(self._parse_and())
        return conditions[0] if len(conditions) == 1 else Or(tuple(conditions))

    def _parse_and(self):
        conditions = [self._parse_unary()]
        while self._accept("AND") is not None:
            conditions.append(self._parse_unary())
        return conditions[0] if len(conditions) == 1 else And(tuple(conditions))

    def _parse_unary(self):
        if self._accept("NOT") is not None:
            return Not(self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self):
        if self._accept("LPAREN") is not None:
            condition = self._parse_or()
            self._expect("RPAREN")
            return condition
        atom = self._expect("ATOM")
        return _condition_from_atom_text(atom.value)


def _condition_from_atom_text(text: str):
    match = _ATOM_RE.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"Invalid ROI image block {text!r}")

    image_text = match.group("image")
    image_index = int(image_text[1:] if image_text.startswith("i") else image_text)
    raw_terms = [term.strip() for term in match.group("terms").split(",")]
    if not raw_terms or any(term == "" for term in raw_terms):
        raise ValueError(
            f"Image block {text!r} must contain one or more ROI selectors like '0' or '!3'"
        )

    conditions = []
    for raw_term in raw_terms:
        roi_match = _ROI_TERM_RE.fullmatch(raw_term)
        if roi_match is None:
            raise ValueError(
                f"Invalid ROI selector {raw_term!r} inside image block {text!r}"
            )
        roi_index = int(roi_match.group("roi"))
        if roi_match.group("neg") is None:
            conditions.append(Occupied(image_index, roi_index))
        else:
            conditions.append(Empty(image_index, roi_index))

    return conditions[0] if len(conditions) == 1 else And(tuple(conditions))


def parse_condition_syntax(text: str):
    """Parse a compact boolean condition expression.

    Example: ``"1[0,1] & 2[!3]"`` means image 1 ROIs 0 and 1 are occupied,
    and image 2 ROI 3 is empty.
    """

    stripped = text.strip()
    if not stripped:
        return Always()
    return _ConditionSyntaxParser(_tokenize_condition_syntax(stripped)).parse()


@dataclass(frozen=True)
class ConditionalBinomialResult:
    """Conditional probability estimates pooled and per group."""

    num_selected_by_group: np.ndarray
    num_successes_by_group: np.ndarray
    probability_by_group: np.ndarray
    probability_error_by_group: np.ndarray
    pooled_num_selected: int
    pooled_num_successes: int
    pooled_probability: float
    pooled_probability_error: float


def _normalise_occupancy_by_image(
    occupancy: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
) -> OccupancyByImage:
    if isinstance(occupancy, np.ndarray):
        occupancy_array = np.asarray(occupancy, dtype=bool)
        if occupancy_array.ndim != 4:
            raise ValueError("occupancy must have shape (shots, image, group, roi)")
        return tuple(
            np.asarray(occupancy_array[:, image_index, :, :], dtype=bool)
            for image_index in range(occupancy_array.shape[1])
        )

    arrays = tuple(np.asarray(image, dtype=bool) for image in occupancy)
    if not arrays:
        raise ValueError("At least one occupancy image array is required")
    if any(array.ndim != 3 for array in arrays):
        raise ValueError("Each occupancy image must have shape (shots, group, roi)")

    first_shots, first_groups, _ = arrays[0].shape
    for array in arrays[1:]:
        shots, groups, _ = array.shape
        if shots != first_shots or groups != first_groups:
            raise ValueError(
                "All occupancy images must share the same number of shots and groups"
            )
    return arrays


def _evaluate_condition_for_group(
    occupancy_by_image: OccupancyByImage,
    condition: Any,
    group_index: int,
) -> np.ndarray:
    if isinstance(condition, Always):
        return np.ones(occupancy_by_image[0].shape[0], dtype=bool)
    if isinstance(condition, Occupied):
        return occupancy_by_image[int(condition.image_index)][
            :,
            int(group_index),
            int(condition.roi_index),
        ]
    if isinstance(condition, Empty):
        return ~occupancy_by_image[int(condition.image_index)][
            :,
            int(group_index),
            int(condition.roi_index),
        ]
    if isinstance(condition, Not):
        return ~_evaluate_condition_for_group(
            occupancy_by_image,
            condition.condition,
            group_index,
        )
    if isinstance(condition, And):
        if not condition.conditions:
            return np.ones(occupancy_by_image[0].shape[0], dtype=bool)
        result = np.ones(occupancy_by_image[0].shape[0], dtype=bool)
        for child in condition.conditions:
            result &= _evaluate_condition_for_group(
                occupancy_by_image,
                child,
                group_index,
            )
        return result
    if isinstance(condition, Or):
        if not condition.conditions:
            return np.zeros(occupancy_by_image[0].shape[0], dtype=bool)
        result = np.zeros(occupancy_by_image[0].shape[0], dtype=bool)
        for child in condition.conditions:
            result |= _evaluate_condition_for_group(
                occupancy_by_image,
                child,
                group_index,
            )
        return result
    raise TypeError(f"Unsupported condition type: {type(condition)!r}")


def conditional_binomial(
    occupancy: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    event,
    given=None,
) -> ConditionalBinomialResult:
    """Return conditional binomial statistics pooled and by group."""

    occupancy_by_image = _normalise_occupancy_by_image(occupancy)
    given_condition = Always() if given is None else given
    num_groups = occupancy_by_image[0].shape[1]

    num_selected_by_group = np.empty(num_groups, dtype=int)
    num_successes_by_group = np.empty(num_groups, dtype=int)
    probability_by_group = np.empty(num_groups, dtype=float)
    probability_error_by_group = np.empty(num_groups, dtype=float)

    for group_index in range(num_groups):
        selected = _evaluate_condition_for_group(
            occupancy_by_image,
            given_condition,
            group_index,
        )
        successes = selected & _evaluate_condition_for_group(
            occupancy_by_image,
            event,
            group_index,
        )
        num_selected = int(np.sum(selected))
        num_successes = int(np.sum(successes))
        probability, probability_error = estimate_probability_from_counts(
            num_successes,
            num_selected,
        )

        num_selected_by_group[group_index] = num_selected
        num_successes_by_group[group_index] = num_successes
        probability_by_group[group_index] = probability
        probability_error_by_group[group_index] = probability_error

    pooled_num_selected = int(np.sum(num_selected_by_group))
    pooled_num_successes = int(np.sum(num_successes_by_group))
    pooled_probability, pooled_probability_error = estimate_probability_from_counts(
        pooled_num_successes,
        pooled_num_selected,
    )

    return ConditionalBinomialResult(
        num_selected_by_group=num_selected_by_group,
        num_successes_by_group=num_successes_by_group,
        probability_by_group=probability_by_group,
        probability_error_by_group=probability_error_by_group,
        pooled_num_selected=pooled_num_selected,
        pooled_num_successes=pooled_num_successes,
        pooled_probability=pooled_probability,
        pooled_probability_error=pooled_probability_error,
    )
