import numpy as np


class OccupancyGrid:

    def __init__(self,
                 frame_width,
                 frame_height,
                 rows=8,
                 cols=8):

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.rows = rows
        self.cols = cols

        self.cell_width = frame_width / cols
        self.cell_height = frame_height / rows

    def build(self, centers):
        """
        centers:
            [(x1,y1),
             (x2,y2),
             ...]

        return:
            numpy array (rows x cols)
        """

        grid = np.zeros(
            (self.rows, self.cols),
            dtype=np.uint8
        )

        for x, y in centers:

            col = int(x / self.cell_width)
            row = int(y / self.cell_height)

            col = min(col, self.cols - 1)
            row = min(row, self.rows - 1)

            grid[row, col] += 1

        return grid