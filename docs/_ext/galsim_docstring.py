"""
Custom Sphinx extension for JAX-GalSim documentation.

Processes docstrings produced by the ``@implements`` decorator.  Each such
docstring contains a ``*Original docstring below.*`` marker that separates
the JAX-specific summary/lax_description from the upstream GalSim text.

This extension:

1. Extracts the ``Parameters:`` section from the original GalSim block and
   re-injects it *before* the collapsible so that Sphinx / Napoleon renders
   the parameters as normal field-list entries.
2. Wraps the rest of the original GalSim narrative in a ``sphinx-design``
   ``.. dropdown::`` directive so it is collapsible in the HTML output.
"""

import re

# The literal text injected by the ``implements`` decorator.
_MARKER = "*Original docstring below.*"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip())


def _parse_galsim_params(lines: list[str]) -> list[tuple[str, str]]:
    """Return ``[(name, description), …]`` parsed from a GalSim Parameters block.

    GalSim parameter entries look like::

        Parameters:
            sigma:              The sigma of the profile.  Typically in arcsec.
                                [One of ``sigma``, ``fwhm``, or ``half_light_radius``.]
            flux:               The flux of the profile.  [default: 1]

    The function tolerates multi-line descriptions and mixed whitespace.
    """
    params: list[tuple[str, str]] = []
    in_params = False
    param_indent: int | None = None
    current_name: str | None = None
    current_desc: list[str] = []

    for line in lines:
        stripped = line.rstrip()

        # Detect the "Parameters:" heading
        if re.match(r"^\s*Parameters\s*:", stripped):
            in_params = True
            param_indent = None
            continue

        if not in_params:
            continue

        # A blank line inside the params block is OK (continuation) but
        # a new top-level heading ends the block.
        if not stripped:
            continue

        indent = _leading_spaces(stripped)

        # A zero-indent non-empty line means we've left the section.
        if indent == 0:
            break

        # First non-empty, non-zero-indent line sets the expected indent level.
        if param_indent is None:
            param_indent = indent

        if indent == param_indent:
            # Save the previous parameter before starting a new one.
            if current_name is not None:
                params.append((current_name, " ".join(current_desc)))
                current_desc = []
            m = re.match(r"^(\w+)\s*:\s*(.*)", stripped.strip())
            if m:
                current_name = m.group(1)
                first_desc = m.group(2).strip()
                if first_desc:
                    current_desc = [first_desc]
            else:
                current_name = None
        elif indent > param_indent and current_name is not None:
            # Continuation of the previous parameter description.
            current_desc.append(stripped.strip())

    # Flush the last parameter.
    if current_name is not None:
        params.append((current_name, " ".join(current_desc)))

    return params


def _remove_params_section(lines: list[str]) -> list[str]:
    """Return *lines* with the ``Parameters:`` block removed."""
    result: list[str] = []
    in_params = False
    param_indent: int | None = None

    for line in lines:
        stripped = line.rstrip()

        if re.match(r"^\s*Parameters\s*:", stripped) and not in_params:
            in_params = True
            param_indent = _leading_spaces(stripped)
            continue

        if in_params:
            if not stripped:
                # Skip blank lines that belong to the params section.
                continue
            indent = _leading_spaces(stripped)
            # Any line at indent ≤ "Parameters:" indent is a new section.
            if indent <= param_indent:
                in_params = False
                result.append(line)
            # else: still inside the params block → skip it.
        else:
            result.append(line)

    return result


# ---------------------------------------------------------------------------
# Main event handler
# ---------------------------------------------------------------------------


def _process_galsim_docstring(
    app, what: str, name: str, obj, options, lines: list[str]
) -> None:
    """``autodoc-process-docstring`` handler."""

    # Find the marker line.
    marker_idx: int | None = None
    for i, line in enumerate(lines):
        if _MARKER in line:
            marker_idx = i
            break

    if marker_idx is None:
        return

    # --- split ---
    jax_lines = lines[:marker_idx]
    original_lines = lines[marker_idx + 1 :]

    # Trim trailing blank lines from the JAX section.
    while jax_lines and not jax_lines[-1].strip():
        jax_lines.pop()

    # Trim leading blank lines from the original section.
    while original_lines and not original_lines[0].strip():
        original_lines.pop(0)

    # --- extract parameters ---
    params = _parse_galsim_params(original_lines)
    original_no_params = _remove_params_section(original_lines)

    # Trim trailing blank lines from the narrative.
    while original_no_params and not original_no_params[-1].strip():
        original_no_params.pop()

    # --- split jax_lines into summary+LAX-ref and lax_description ---
    # The @implements decorator always injects "LAX-backend implementation of …"
    # as the second paragraph.  Any content after that line is lax_description.
    lax_ref_idx: int | None = None
    for i, line in enumerate(jax_lines):
        if "LAX-backend implementation of" in line:
            lax_ref_idx = i
            break

    if lax_ref_idx is not None:
        header_lines = jax_lines[: lax_ref_idx + 1]
        desc_lines = jax_lines[lax_ref_idx + 1 :]
        # Strip surrounding blank lines from the description block.
        while desc_lines and not desc_lines[0].strip():
            desc_lines.pop(0)
        while desc_lines and not desc_lines[-1].strip():
            desc_lines.pop()
    else:
        header_lines = jax_lines
        desc_lines = []

    # --- build the replacement lines ---
    new_lines: list[str] = list(header_lines)
    new_lines.append("")

    # Wrap lax_description in a Sharp Bits admonition when present.
    if desc_lines:
        new_lines.append(
            ".. admonition:: \U0001f52a JAX-GalSim - The Sharp Bits \U0001f52a"
        )
        new_lines.append("   :class: warning")
        new_lines.append("")
        for line in desc_lines:
            if line.strip():
                new_lines.append("   " + line)
            else:
                new_lines.append("")
        new_lines.append("")

    # Inject parameters in Google style so Napoleon renders them properly.
    if params:
        new_lines.append("Parameters:")
        for pname, pdesc in params:
            # Use 4-space indent, which Napoleon / Google style expects.
            new_lines.append(f"    {pname}: {pdesc}")
        new_lines.append("")

    # Wrap the original narrative in a collapsible dropdown.
    has_content = any(line.strip() for line in original_no_params)
    if has_content:
        new_lines.append(".. dropdown:: Original GalSim Documentation")
        new_lines.append("   :class-container: sd-shadow-sm")
        new_lines.append("   :color: secondary")
        new_lines.append("")
        for line in original_no_params:
            if line.strip():
                new_lines.append("   " + line)
            else:
                new_lines.append("")
        new_lines.append("")

    lines[:] = new_lines


# ---------------------------------------------------------------------------
# Extension setup
# ---------------------------------------------------------------------------


def setup(app):
    app.connect("autodoc-process-docstring", _process_galsim_docstring)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
