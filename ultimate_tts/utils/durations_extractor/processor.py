from pathlib import Path
from textgrid import TextGrid


class DurationsProcessor:
    def __init__(self):
        return

    def process_files(self, inputs, outputs, verbose=False):
        input_textgrids = Path(inputs["textgrids_path"])
        output_durations_path = Path(outputs["durations_path"])
        

        for textgrid_file_path in input_textgrids.glob("*.TextGrid"):
            text_grid = TextGrid.fromFile(str(textgrid_file_path))

            phones_intervals = None
            
            for intervals in text_grid:
                if intervals.name == "phones":
                    phones_intervals = intervals
                    break

            assert intervals is not None, f"Phones intervals not found in textgrid file {str(textgrid_file_path)}"
