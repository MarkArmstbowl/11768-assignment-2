"""Capture a replay's Matplotlib content, shared axes, and figure-level labels.

Extends the staff-provided capture module without importing the test assertions.
The saved PNG is untouched by this hook.
"""
try:
    import figure_manifest as _capture

    _base_manifest = _capture.manifest_from_figure

    def _manifest_with_scope(fig):
        result = _base_manifest(fig)
        result["figure_texts"] = []
        for artist in fig.texts:
            # Coordinates are normalized to the whole figure, not to an axes.
            point = fig.transFigure.inverted().transform(
                artist.get_transform().transform(artist.get_position())
            )
            result["figure_texts"].append({
                "text": artist.get_text(),
                "position": [float(point[0]), float(point[1])],
                "visible": bool(artist.get_visible()),
                "rotation": float(artist.get_rotation()),
            })
        for index, ax in enumerate(fig.axes):
            result["axes"][index]["xticks"] = [float(value) for value in ax.get_xticks()]
            siblings = ax.get_shared_x_axes().get_siblings(ax)
            result["axes"][index]["shares_x_with"] = [
                other_index for other_index, other in enumerate(fig.axes)
                if other is not ax and other in siblings
            ]
        return result

    _capture.manifest_from_figure = _manifest_with_scope
    _capture.install_savefig_hook()
except Exception:
    # Missing capture fails closed when the verifier requires its manifest.
    pass
