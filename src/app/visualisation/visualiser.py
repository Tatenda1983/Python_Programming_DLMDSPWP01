from bokeh.io import output_file
from bokeh.layouts import column, row, gridplot
from bokeh.models import Div, ColumnDataSource
from bokeh.palettes import Category10
from bokeh.plotting import figure, save


class Visualiser:
    """Produces a multi-panel Bokeh visualisation."""

    COLOURS = Category10[10]

    def __init__(
        self,
        train_df,
        ideal_df,
        results_df,
        chosen
    ):
        self.train = train_df
        self.ideal = ideal_df
        self.results = results_df
        self.chosen = chosen

    def _train_vs_ideal_panels(self):
        """Create one figure for each training/ideal function pair."""

        panels = []

        for train_col, ideal_col in self.chosen.items():

            p = figure(
                width=380,
                height=280,
                title=f"Train {train_col} -> Ideal {ideal_col}",
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )

            p.line(
                self.train["x"],
                self.train[train_col],
                line_color=self.COLOURS[0],
                line_width=2,
                legend_label=f"Train {train_col}"
            )

            p.line(
                self.ideal["x"],
                self.ideal[ideal_col],
                line_color=self.COLOURS[1],
                line_width=1.5,
                line_dash="dashed",
                legend_label=f"Ideal {ideal_col}"
            )

            p.legend.label_text_font_size = "7pt"
            p.legend.location = "top_left"
            p.legend.background_fill_alpha = 0.6

            panels.append(p)

        return panels

    def _summary_div(self):
        """Create a summary panel showing the mapping results."""

        matched = self.results["ideal_func_no"].notna().sum()

        lines = [
            "<b>Chosen Ideal Functions</b>",
            "<hr style='margin:4px 0'>"
        ]

        for train_col, ideal_col in self.chosen.items():
            lines.append(
                f"{train_col} &rarr; {ideal_col}"
            )

        lines.append("<br>")
        lines.append(f"Test points : {len(self.results)}")
        lines.append(f"Matched : {matched}")
        lines.append(
            f"Unmatched : {len(self.results) - matched}"
        )

        html = (
            "<div style='font-family:monospace; "
            "font-size:13px; "
            "background:#f0f0f0; "
            "border:1px solid #aaa; "
            "border-radius:6px; "
            "padding:12px; "
            "width:340px;'>"
            + "<br>".join(lines)
            + "</div>"
        )

        return Div(
            text=html,
            width=340,
            height=280
        )

    def _test_mapping_plot(self):
        """Create the test-point mapping plot."""

        p = figure(
            width=1180,
            height=420,
            title="Test Data Mapping to Chosen Ideal Functions",
            x_axis_label="x",
            y_axis_label="y",
            tools="pan,wheel_zoom,box_zoom,reset,save,hover",
            tooltips=[
                ("x", "@x"),
                ("y", "@y"),
                ("delta", "@delta_y"),
                ("ideal fn", "@ideal_func_no")
            ]
        )

        ideal_cols = list(self.chosen.values())

        col_map = {
            ideal_col: self.COLOURS[index % len(self.COLOURS)]
            for index, ideal_col in enumerate(ideal_cols)
        }

        for ideal_col in ideal_cols:

            p.line(
                self.ideal["x"],
                self.ideal[ideal_col],
                line_color=col_map[ideal_col],
                line_width=1.2,
                line_dash="dashed",
                line_alpha=0.4
            )

            subset = self.results[
                self.results["ideal_func_no"] == ideal_col
            ]

            source = ColumnDataSource(subset)

            p.scatter(
                "x",
                "y",
                source=source,
                size=8,
                color=col_map[ideal_col],
                legend_label=f"-> {ideal_col}"
            )

        unmatched = self.results[
            self.results["ideal_func_no"].isna()
        ]

        if not unmatched.empty:

            source_unmatched = ColumnDataSource(unmatched)

            p.scatter(
                "x",
                "y",
                source=source_unmatched,
                size=8,
                marker="x",
                color="grey",
                legend_label="Unmatched"
            )

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        return p

    def plot(self, save_path):
        """Build and save the complete visualisation."""

        output_file(
            save_path,
            title="Python Assignment - Results"
        )

        top_grid = gridplot(
            self._train_vs_ideal_panels(),
            ncols=2
        )

        top_row = row(
            top_grid,
            self._summary_div()
        )

        bottom = self._test_mapping_plot()

        title_div = Div(
            text="<h2>Python Assignment - Results</h2>"
        )

        layout = column(
            title_div,
            top_row,
            bottom
        )

        save(layout)

        print(
            f"  Visualisation saved -> {save_path}"
        )