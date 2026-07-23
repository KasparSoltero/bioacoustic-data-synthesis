# synthesiser/interactive.py
# Interactive review widget for the synthesis pipeline. Lets you step through
# generated soundscapes one at a time, toggle box/mask overlays, and play audio.

import matplotlib.pyplot as plt
from synthesiser.visualisation import plot_spectrogram

HELP_TEXT = "[n] next  [b] toggle boxes  [m] toggle masks  [o] toggle spans  [space] play audio  [q] quit"


def review_sample(waveform, spec, annotations, idx, n_total):
    """
    Blocks until the user presses a hotkey that advances or quits.
    Returns 'next' or 'quit'.
    """
    state = {'show_boxes': True, 'show_masks': True, 'show_spans': True, 'action': 'next'}
    fig, ax = plt.subplots(figsize=(10, 4))

    def _filtered_annotations():
        filtered = []
        for ann in annotations:
            a = dict(ann)
            if not state['show_boxes']:
                a['box'] = None
            if not state['show_masks']:
                a['mask'] = None
            filtered.append(a)
        return filtered

    def redraw():
        ax.clear()
        plot_spectrogram(
            spec, ax=ax, show=False, db_scale=True, cmap='dusk',
            annotations=_filtered_annotations(),
            show_spans=state['show_spans'],
        )
        ax.set_title(
            f"Sample {idx + 1}/{n_total}   "
            f"boxes:{'on' if state['show_boxes'] else 'off'}  "
            f"masks:{'on' if state['show_masks'] else 'off'}  "
            f"spans:{'on' if state['show_spans'] else 'off'}\n{HELP_TEXT}",
            fontsize=9,
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'n':
            state['action'] = 'next'
            plt.close(fig)
        elif event.key == 'q':
            state['action'] = 'quit'
            plt.close(fig)
        elif event.key == 'b':
            state['show_boxes'] = not state['show_boxes']
            redraw()
        elif event.key == 'm':
            state['show_masks'] = not state['show_masks']
            redraw()
        elif event.key == 'o':
            state['show_spans'] = not state['show_spans']
            redraw()
        elif event.key == ' ':
            waveform.play()

    fig.canvas.mpl_connect('key_press_event', on_key)
    redraw()
    plt.show()

    return state['action']