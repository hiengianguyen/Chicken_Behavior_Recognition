class TrajectorySmoother:

    def __init__(self, alpha=0.3):

        self.alpha = alpha

    def smooth_track(self, track_df):

        track_df = track_df.sort_values("frame").copy()

        smooth_x = []
        smooth_y = []

        prev_x = None
        prev_y = None

        for _, row in track_df.iterrows():

            x = row["x"]
            y = row["y"]

            if prev_x is None:

                sx = x
                sy = y

            else:

                sx = self.alpha * x + (1 - self.alpha) * prev_x
                sy = self.alpha * y + (1 - self.alpha) * prev_y

            smooth_x.append(sx)
            smooth_y.append(sy)

            prev_x = sx
            prev_y = sy

        track_df["smooth_x"] = smooth_x
        track_df["smooth_y"] = smooth_y

        return track_df