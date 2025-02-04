from nbconvert.preprocessors import Preprocessor


class HideSectionPreprocessor(Preprocessor):
    """
    Removes all cells from a certain heading (identified by text or a special tag)
    until the next heading of the same or higher level.
    """

    def preprocess(self, nb, resources):
        # For simplicity, define the heading text that triggers hiding:
        heading_to_hide = "Hidden Section"

        removing = False
        new_cells = []

        for cell in nb.cells:
            # Check if this is a markdown heading
            if cell.cell_type == "markdown":
                lines = cell.source.splitlines()
                if lines:
                    # simple check for a markdown heading line
                    first_line = lines[0].lstrip()
                    # e.g., # Hidden Section   ->  1st level heading
                    # or '## Hidden Section'  ->  2nd level, etc.
                    if first_line.startswith("#"):
                        # Is it the heading we want to hide or do we start showing again?
                        if heading_to_hide in first_line:
                            removing = True
                        else:
                            # If we see a new heading, stop removing
                            removing = False

            # If we're not in remove mode, keep the cell
            if not removing:
                new_cells.append(cell)

        nb.cells = new_cells
        return nb, resources
