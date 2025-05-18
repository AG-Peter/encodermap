# -*- coding: utf-8 -*-
# encodermap/plotting/dashboard.py
################################################################################
# EncoderMap: A python library for dimensionality reduction.
#
# Copyright 2019-2024 University of Konstanz and the Authors
#
# Authors:
# Kevin Sawade
#
# Encodermap is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
# This package is distributed in the hope that it will be useful to other
# researches. IT DOES NOT COME WITH ANY WARRANTY WHATSOEVER; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# See <http://www.gnu.org/licenses/>.
################################################################################
"""EncoderMap's dashboard. Explore and understand your MD data.

"""


################################################################################
# Imports
################################################################################


# Future Imports at the top
from __future__ import annotations

# Standard Library Imports
import builtins
import json
import os
import random
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, Optional, Union

# Third Party Imports
import dash
import dash_auth
import dash_bio as dashbio
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import flask
import plotly.graph_objs as go
from dash import Dash, Input, Output, State, callback, ctx, dcc, html
from dash.exceptions import PreventUpdate
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url

# Encodermap imports
from encodermap import load_project
from encodermap.kondata import get_assign_from_file
from encodermap.plot.interactive_plotting import InteractivePlotting
from encodermap.plot.plotting import _plot_ball_and_stick


################################################################################
# Typing
################################################################################


if TYPE_CHECKING:
    # Third Party Imports
    import mdtraj as md


################################################################################
# Globals
################################################################################


################################################################################
# Helpers
################################################################################


def _redefine_open() -> None:
    """Redefines the `open()` builtin to trick MDTraj to use `ReplaceOpenWithStringIO`.

    MDTraj can't save to a StringIO object, because it checks the str provided
    in the filename argument for the extension and decides the format based on that
    (even using the `save_pdb()` function checks the extension). As StringIO objects
    can't be used like a string this function together with `_redefine_os_path_exists`
    and `ReplaceOpenWithStringIO` are used to trick MDtraj.

    """
    orig_func = builtins.open

    def new_func(*args, **kwargs):
        if str(args[0].lower()) == "stringio.pdb":
            return args[0]
        else:
            return orig_func(*args, **kwargs)

    builtins.open = new_func


def _redefine_os_path_exists() -> None:
    """Redefines the os.path.exists() function to trick MDTRaj to use `ReplaceOpenWithStringIO`"""
    orig_func = os.path.exists

    def new_func(path):
        if path.lower() == "stringio.pdb":
            return True
        else:
            return orig_func(path)

    os.path.exists = new_func


class ReplaceOpenWithStringIO:
    """Tricks MDTraj to write the output into a StringIO object.

    Inside a context-manager, this class will redefine the builtin `oprn()`
    function and overwrite the `os.path.exists()` function to use it as a
    sort-of str object.

    """

    def __init__(self):
        self.stringio = StringIO()
        self._orig_open = builtins.open
        self._orig_path_exists = os.path.exists

    def __enter__(self):
        _redefine_open()
        _redefine_os_path_exists()
        return self

    def __exit__(self, type, value, traceback):
        builtins.open = self._orig_open
        os.path.exists = self._orig_path_exists
        self.stringio.seek(0)

    def write(self, *args, **kwargs) -> None:
        """Write into the StringIO object."""
        self.stringio.write(*args, **kwargs)

    def read(self) -> str:
        """Read from the StringIO object."""
        return self.stringio.read()

    def lower(self) -> str:
        """Functions will think, this is a str's builtin `lower()` function."""
        return "stringio.pdb"

    def __str__(self):
        return "stringio.pdb"

    def close(self):
        pass


def traj_to_pdb(
    traj: md.Trajectory,
) -> list[dict[str, Union[str, bool, dict[str, str]]]]:
    """Converts an MDTraj Trajectory into a dict, that can be understood by
    dashbio's NglMolViewer.

    Args:
        traj (md.Trajectory): The MDTraj trajectory.

    Returns:
         list[dict[str, Union[str, bool, dict[str, str]]]]: The json-type data for
            dashbio's NglMolViewer.

    """
    with ReplaceOpenWithStringIO() as r:
        traj.save_pdb(r)
    randname = f"{random.getrandbits(128):032x}"
    pdb_content = r.read()

    # manually fix by adding peptide bonds
    lines = pdb_content.splitlines()
    # g = traj.top.to_bondgraph()
    # for node in g.nodes:
    #     neighbors = " ".join([f"{n.index + 1:>4}" for n in g.neighbors(node)])
    #     lines.insert(-2, f"CONECT {node.index + 1:>4} {neighbors}")
    for chain in traj.top.chains:
        residues = [r for r in chain.residues]
        for r1, r2 in zip(residues[:-1], residues[1:]):
            a1 = {a.name: a.index + 1 for a in r1.atoms}
            a2 = {a.name: a.index + 1 for a in r2.atoms}
            lines.insert(-2, f"CONECT {a1['C']:>4} {a2['N']:>4}")
    pdb_content = "\n".join(lines)
    data_list = [
        {
            "filename": f"{randname}.pdb",
            "ext": "pdb",
            "selectedValue": f"{randname[:4].upper()}",
            "chain": "ALL",
            "aaRange": "ALL",
            "chosen": {"atoms": "", "residues": ""},
            "color": "sstruc",
            "config": {"type": "text/plain", "input": pdb_content},
            "resetView": True,
            "uploaded": False,
        }
    ]
    return data_list


################################################################################
# App
################################################################################


class DebugPage:
    def __init__(self):
        self.display = html.Code(id="debug-display")
        dash.register_page("debug", layout=self.layout)

    @property
    def layout(self):
        layout = html.Div(
            [
                html.H1("Debug Page"),
                self.text_area,
                self.display,
            ]
        )
        return layout

    @property
    def text_area(self):
        text_input = html.Div(
            [
                dbc.Alert(
                    "Keep in mind, that this webpage is bad practice. It can "
                    "execute arbitrary code. If you see this page on a deployed "
                    "webpage, kill the server immediately. Otherwise use this "
                    "page to access info about the running dash app. "
                    "Enter code to evaluate. Accept with Enter-key...",
                    color="danger",
                ),
                dbc.Textarea(
                    placeholder='print("Hello World!")',
                    rows=10,
                    id="debug-textarea",
                ),
            ]
        )
        return text_input

    @staticmethod
    @callback(
        Output("debug-display", "children"),
        Input("debug-textarea", "n_submit"),
        State("debug-textarea", "value"),
        prevent_initial_call=True,
    )
    def run_code(value, state):
        try:
            out = str(eval(state))
        except Exception as e:
            out = str(e)
        return out


class LocalUploadTraj:
    _placeholder = """\
    trajs = ["/path/to/traj1.xtc", "/path/to/traj2.xtc"]
    tops = ["/path/to/traj1.pdb", "/path/to/traj2.pdb"]
    common_str = ["traj1, "traj2"]
    """

    def __init__(self, main):
        self.main = main

        # the display
        self.display = dbc.Container(
            [dbc.Card([], id="upload-display", class_name="align-items-center")],
            fluid=True,
        )

        # the upload area
        self.upload_card_body = dbc.CardBody(
            [
                html.H4("Upload files"),
                dcc.Upload(
                    id="upload-data-input",
                    children=html.Div(
                        ["Drag/Drop or ", html.A("Select Files")],
                        className="h-100",
                    ),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px",
                    },
                    # Allow multiple files to be uploaded
                    multiple=True,
                ),
            ]
        )
        self.upload_card = dbc.Col(
            [
                dbc.Card(
                    self.upload_card_body,
                    id="upload-data-input-card",
                    class_name="h-100",
                )
            ],
            width=3,
            class_name="col-sm-6 col-lg-3",
        )

        # the textarea
        self.text_area_card_body = dbc.CardBody(
            [
                html.H4("Local files"),
                dbc.Textarea(
                    placeholder=self._placeholder,
                    rows=3,
                    readonly=False,
                    id="upload-paths-input",
                    class_name="h-50",
                ),
            ],
        )
        self.text_area_card = dbc.Col(
            [
                dbc.Card(
                    self.text_area_card_body,
                    id="upload-paths-card",
                    class_name="h-100",
                ),
            ],
            width=6,
            class_name="col-sm-6 col-lg-6",
        )

        # the project area
        self.input_card_body = dbc.CardBody(
            [
                html.H4("Project"),
                dbc.Input(
                    placeholder="linear_dimers",
                    id="upload-project-input",
                    class_name="h-50",
                ),
            ],
        )
        self.input_card = dbc.Col(
            [
                dbc.Card(
                    self.input_card_body,
                    id="upload-project-card",
                    class_name="h-100",
                ),
            ],
            width=3,
            class_name="col-sm-6 col-lg-3",
        )

        # the complete container
        self.upload_container = dbc.Container(
            [
                html.Div(
                    [
                        self.upload_card,
                        self.text_area_card,
                        self.input_card,
                    ],
                    id="upload-hide",
                    className="row",
                    style={"display": "none", "height": "30%"},
                ),
                html.Div(style={"margin-top": "20px"}),
                dbc.Row(
                    [
                        dbc.Button(
                            "Upload",
                            id="upload-button",
                            style={"width": "95%", "margin": "auto"},
                        )
                    ]
                ),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Button(
                            "Linear Dimers Project",
                            id="linear-dimers-button",
                            style={"width": "95%", "margin": "auto"},
                        )
                    ]
                ),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Button(
                            "1am7 Project",
                            id="1am7-button",
                            style={"width": "95%", "margin": "auto"},
                        )
                    ]
                ),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Button(
                            "Reset",
                            id="upload-reset-button",
                            style={"width": "95%", "margin": "auto"},
                        )
                    ]
                ),
            ],
            fluid=True,
            style={"height": "75vh"},
        )

        # define the layout
        self.layout = html.Div(
            [
                self.display,
                self.main.store,
                html.Br(),
                dmc.LoadingOverlay(
                    self.upload_container,
                ),
            ],
            style={"margin": "2%"},
        )

        # define local callbacks
        self.main.app.callback(
            Output("upload-display", "children"),
            Output("upload-hide", "style"),
            Output("main-store", "data"),
            Input("upload-button", "n_clicks"),
            Input("linear-dimers-button", "n_clicks"),
            Input("1am7-button", "n_clicks"),
            Input("upload-reset-button", "n_clicks"),
            State("main-store", "data"),
            State("upload-data-input", "contents"),
            State("upload-data-input", "filename"),
            State("upload-data-input", "last_modified"),
            State("upload-paths-input", "value"),
            State("upload-project-input", "value"),
        )(self.load_trajs)

        # register the pae
        dash.register_page("upload", layout=self.layout)

    def load_trajs(
        self,
        upload_n_clicks,  # Input
        reset_n_clicks,  # Input
        linear_dimers_n_clicks,  # Input
        n_click_1am7,  # Input
        main_store,  # State
        list_of_contents,  # State
        list_of_names,  # State
        list_of_dates,  # State
        textarea_value,  # State
        project_value,  # State
    ):
        if main_store is None:
            main_store = {}
        triggered_id = ctx.triggered_id

        # reset button pressed
        if triggered_id == "upload-reset-button":
            if hasattr(self.main, "trajs"):
                del self.main.trajs
            return (
                dbc.CardBody(f"Session was reset. Choose MD data to upload."),
                {"height": "30%"},
                {},
            )

        if triggered_id == "linear-dimers-button":
            if hasattr(self.main, "trajs"):
                del self.main.trajs
            if main_store is None:
                main_store = {}
            main_store["traj_type"] = "project"
            main_store["traj"] = "linear_dimers"
            self.main.traj_page.parse_trajs(main_store)
            return (
                dbc.CardBody(f"Data loaded. Press 'Reset' to reset the session."),
                {"display": "none"},
                main_store,
            )
        if triggered_id == "1am7-button":
            if hasattr(self.main, "trajs"):
                del self.main.trajs
            if main_store is None:
                main_store = {}
            main_store["traj_type"] = "project"
            main_store["traj"] = "1am7"
            self.main.traj_page.parse_trajs(main_store)
            return (
                dbc.CardBody(f"Data loaded. Press 'Reset' to reset the session."),
                {"display": "none"},
                main_store,
            )

        empty = [
            dbc.CardBody(f"Data loaded. Press 'Reset' to reset the session."),
            {"display": "none"},
            main_store,
        ]

        if triggered_id is None:
            if isinstance(main_store, dict):
                if "traj_type" in main_store:
                    return tuple(empty)
            return (
                dbc.CardBody(f"Choose MD data to upload."),
                {"height": "30%"},
                main_store,
            )

        # upload button pressed
        uploaded_any = (
            list_of_contents is not None
            or textarea_value is not None
            or project_value is not None
        )
        if triggered_id == "upload-button" and not uploaded_any:
            return (
                dbc.CardBody(
                    f"Place files in the upload window, or enter local files, "
                    f"or a project name before pressing upload."
                ),
                {"height": "30%"},
                main_store,
            )

        # here we transform data
        if list_of_contents is not None:
            main_store.update({"traj_type": "paths", "traj": list_of_contents})
            # self.main.traj_page.parse_trajs(main_store)
            empty[0] = dbc.CardBody(f"Uploading files currently not implemented..")
        elif textarea_value is not None:
            main_store.update({"traj_type": "text", "traj": textarea_value})
            self.main.traj_page.parse_trajs(main_store)
            empty[0] = dbc.CardBody(
                f"Loading trajectories. Go to the 'Traj' page to look at your data."
            )
        elif project_value is not None:
            main_store.update({"traj_type": "project", "traj": project_value})
            self.main.traj_page.parse_trajs(main_store)
            empty[0] = dbc.CardBody(
                f"Loading project '{project_value}'. Go to the 'Traj' page to look at your data."
            )

        empty[-1] = main_store
        return tuple(empty)


class TopPage:
    def __init__(self, main):
        self.main = main

        # the display
        self.display = dbc.Container(
            [dbc.Card([], id="top-display", class_name="align-items-center")],
            fluid=True,
        )

        # the topology dropdown selector
        self.topology_container = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                id="top-dynamic-dropdown",
                                style={"width": "100%"},
                                persistence=True,
                                persistence_type="session",
                            ),
                            width=3,
                            class_name="col-sm-6 col-lg-3",
                        ),
                        dbc.Col(
                            dbc.RadioItems(
                                options=[
                                    {"label": "Atoms", "value": 0},
                                    {"label": "Bonds", "value": 1},
                                    {"label": "Angles", "value": 2},
                                    {"label": "Dihedrals", "value": 3},
                                ],
                                inline=True,
                                persistence=True,
                                persistence_type="session",
                                value=0,
                                id="top-radioitems-input",
                            ),
                            width=5,
                            class_name="col-sm-6 col-lg-4",
                        ),
                        dbc.Col(
                            dbc.Row(
                                [
                                    html.P("Atom Subset"),
                                    dcc.RangeSlider(
                                        0,
                                        0,
                                        value=[0, 0],
                                        id="top-atom-indices-rangeslider",
                                        persistence=True,
                                        persistence_type="session",
                                    ),
                                ],
                            ),
                            width=4,
                            class_name="col-sm-6 col-lg-4",
                        ),
                    ],
                    style={"width": "100%"},
                ),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        dmc.JsonInput(
                                            label="Custom Amino Acids:",
                                            value="{}",
                                            autosize=True,
                                            minRows=30,
                                            id="top-json-input",
                                        ),
                                    ],
                                ),
                                html.Br(),
                                dbc.Button(
                                    "Load",
                                    id="top-json-load",
                                    style={"width": "95%", "margin": "auto"},
                                ),
                            ],
                            id="top-custom-aas",
                            width=3,
                            class_name="col-sm-6 col-lg-3",
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [],
                                    id="top-top-plot",
                                    style={"height": "800px"},
                                ),
                            ],
                            width=9,
                            class_name="col-sm-6, col-lg-9",
                        ),
                    ],
                ),
            ],
            style={"display": "none", "height": "85%"},
            id="top-hide",
            fluid=True,
        )

        # the dummy div, that is used to run the context when the page is accessed
        self.dummy_div = html.Div([], id="top-page-dummy-div")

        # define the layout
        self.layout = html.Div(
            [
                self.display,
                self.main.store,
                html.Br(),
                dmc.LoadingOverlay(
                    self.topology_container,
                ),
                self.dummy_div,
            ],
            style={"margin": "2%"},
        )

        # decorate the interactiveness of the page
        self.main.app.callback(
            Output("top-display", "children"),
            Output("top-hide", "style"),
            Output("top-dynamic-dropdown", "options"),
            State("top-json-input", "value"),
            State("top-dynamic-dropdown", "value"),
            Input("top-json-load", "n_clicks"),
            Input("main-store", "data"),
            Input("top-page-dummy-div", "children"),
        )(self.display_top)

        # display the custom amino acids of the selected topology as json
        self.main.app.callback(
            Output("top-json-input", "value"),
            Output("top-atom-indices-rangeslider", "max"),
            Output("top-atom-indices-rangeslider", "value"),
            Input("top-dynamic-dropdown", "value"),
        )(self.display_custom_aas)

        # display the topology
        self.main.app.callback(
            Output("top-top-plot", "children"),
            State("top-dynamic-dropdown", "value"),
            Input("top-radioitems-input", "value"),
            Input("top-page-dummy-div", "children"),
            Input("top-atom-indices-rangeslider", "value"),
        )(self.display_plot)

        # register the page
        dash.register_page(
            "top",
            layout=self.layout,
        )

    def display_plot(
        self,
        top_value,
        radio_value,
        dummy,
        rangeslider_value,
    ) -> Any:
        if top_value is None:
            top_value = 0

        if not hasattr(self.main, "trajs"):
            raise PreventUpdate

        highlight: Literal["atoms", "bonds", "angles", "dihedrals"] = "atoms"
        if radio_value == 0:
            pass
        elif radio_value == 1:
            highlight = "bonds"
        elif radio_value == 2:
            highlight = "angles"
        elif radio_value == 3:
            highlight = "dihedrals"

        if rangeslider_value != [0, 0]:
            atom_indices = list(range(rangeslider_value[0], rangeslider_value[1]))
        else:
            atom_indices = None

        top = self.main.trajs.top[top_value]
        traj = self.main.trajs.trajs_by_top[top][0]

        try:
            fig = _plot_ball_and_stick(
                traj,
                highlight=highlight,
                atom_indices=atom_indices,
            )
        except Exception as e:
            raise Exception(f"{atom_indices=}") from e
        self.main._figures.append(fig)
        return dcc.Graph(
            figure=fig,
        )

        return f"Pressed {radio_value=} {top_value=}"

    def display_custom_aas(self, top):
        if not hasattr(self.main, "trajs"):
            raise PreventUpdate
        if top is None:
            return "{}", 0, [0, 0]
        top = self.main.trajs.top[top]
        trajs = self.main.trajs.trajs_by_top[top]
        custom_aas = [t._custom_top for t in trajs]
        if len(custom_aas) > 1:
            if any([custom_aas[0] != c for c in custom_aas[1:]]):
                msg = f"The trajectories contain different `_custom_aas`. I am not able to display them:"
                for t in trajs:
                    msg += f"\n{t.basename}:\n{t._custom_top.to_json()}"
                return msg
        return custom_aas[0].to_json(), top.n_atoms, [0, top.n_atoms]

    def get_options(self):
        options = []
        for i, top in enumerate(self.main.trajs.top):
            top_str = str(top).lstrip("<mdtraj.Topology with ").rstrip(">")
            options.append({"label": top_str, "value": i})
        return options

    def fill_dropdown(self, main_store, search_value):
        if main_store is None:
            raise PreventUpdate
        if "traj_type" not in main_store:
            raise PreventUpdate
        if not hasattr(self.main, "trajs"):
            raise PreventUpdate
        return self.get_options()

    def display_top(self, json_values, top_value, n_clicks, main_store, dummy):
        blank_text = f"View and modify topologies on this page after you upload them."
        empty = (dbc.CardBody(blank_text), {"display": "none"}, {})
        if main_store is None:
            return empty
        if "traj_type" not in main_store:
            return empty
        if not hasattr(self.main, "trajs"):
            self.main.traj_page.parse_trajs(main_store)

        triggered_id = ctx.triggered_id
        if triggered_id == "top-json-load":
            if json_values == "{}":
                return (
                    dbc.CardBody(
                        f"Provide custom amino-acids for the selected topology "
                        f"{self.main.trajs.trajs_by_top[self.main.trajs.top[top_value]]} "
                        f"to load."
                    ),
                    {"height": "85%"},
                    self.get_options(),
                )
            else:
                try:
                    data = json.loads(json_values)
                except json.decoder.JSONDecodeError as e:
                    return (
                        dbc.CardBody(
                            f"Couldn't parse your json. I got the error: {e}."
                        ),
                        {"height": "85%"},
                        self.get_options(),
                    )
            try:
                self.main.trajs.trajs_by_top[
                    self.main.trajs.top[top_value]
                ].load_custom_topology(data)
            except Exception as e:
                return (
                    dbc.CardBody(f"Couldn't load the custom topology: {e}"),
                    {"height": "85%"},
                    self.get_options(),
                )
            return (
                dbc.CardBody(f"Custom topology loaded."),
                {"height": "85%"},
                self.get_options(),
            )

        return (
            dbc.CardBody(f"Topologies are available: {self.main.trajs.top=}"),
            {"height": "85%"},
            self.get_options(),
        )


class TrajPage:  # pragma: no doccheck
    def __init__(self, main):
        self.main = main
        self.decorated = False
        self.main_div = html.Div([], id="traj-page-div")
        self.dummy_div = html.Div([], id="traj-page-dummy-div")
        dash.register_page(
            "traj",
            layout=html.Div(
                [
                    self.main.store,
                    self.main_div,
                    self.dummy_div,
                ]
            ),
        )

        self.main.app.callback(
            Output("traj-page-div", "children"),
            Input("main-store", "data"),
            Input("traj-page-dummy-div", "children"),
        )(self.display_trajs)

    def display_trajs(
        self,
        main_store,
        dummy,
    ):
        if main_store is None:
            return html.P(f"Trajs will appear here after you upload them.")
        if "traj_type" not in main_store:
            return html.P(f"Trajs will appear here after you upload them.")

        if not hasattr(self.main, "trajs"):
            self.parse_trajs(main_store)

        return self.traj_loaded_layout(-1)

    def display_traj(self, *args, **kwargs):
        triggered_id = ctx.triggered_id
        return html.Div(html.P(f"{args=} {kwargs=} {triggered_id=}"))

    def traj_loaded_layout(self, traj_num: int = -1):
        if traj_num == -1:
            layout = html.Div(
                [
                    dbc.Row(
                        [
                            dbc.DropdownMenu(
                                label="Trajectory number",
                                id="traj-page-dropdown",
                                children=[
                                    dbc.DropdownMenuItem(
                                        f"traj: {traj.traj_num} {traj.common_str}",
                                        id=f"traj-page-dropdown-item-{traj.traj_num}",
                                    )
                                    for traj in self.main.trajs
                                ],
                            )
                        ]
                    ),
                    html.P(f"Here will trajs be displayed {self.main.trajs}"),
                    html.Div("This is where the custom React component should go."),
                ]
            )

        # add the dynamic callback to display_traj
        if not self.decorated:
            self.main.app.callback(
                Output("traj-page-display", "children"),
                [
                    Input(f"traj-page-dropdown-item-{traj.traj_num}", "n_clicks")
                    for traj in self.main.trajs
                ],
                prevent_initial_call=True,
            )(self.display_traj)
            self.decorated = True

        return layout

    def parse_trajs(self, data):
        if data["traj_type"] == "project":
            self.main.trajs, self.main.autoencoder = load_project(
                data["traj"], load_autoencoder=True
            )
        else:
            raise NotImplementedError


class ProjectionPage(InteractivePlotting):
    def __init__(self, main):
        self.main = main
        self.scatter = None

        # the display
        self.display = dbc.Container(
            [
                dbc.Card(
                    [html.P("Go to the 'Load' page to load data.")],
                    id="projection-page-display",
                    class_name="align-items-center",
                ),
            ],
            fluid=True,
        )

        # the topology dropdown selector
        self.figure_widget = go.FigureWidget()
        self.trace_widget = go.FigureWidget()
        self.projection_container = dbc.Container(
            [
                dbc.Row(),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                figure=self.figure_widget, id="projection-page-plot"
                            ),
                            style={"height": "40vh"},
                            class_name="col-lg-6",
                        ),
                        dbc.Col(
                            dcc.Graph(
                                figure=self.trace_widget, id="projection-page-trace"
                            ),
                            style={"height": "40vh"},
                            class_name="col-lg-1",
                        ),
                        dbc.Col(
                            html.Div([], id="ngl-container", style={"height": "40vh"}),
                            style={"height": "40vh"},
                            class_name="col-lg-5",
                            id="projection-page-view",
                        ),
                    ],
                    style={"width": "100%", "height": "40%"},
                ),
            ],
            style={"height": "85%"},
            id="projection-page-hide",
            fluid=True,
        )

        # the dummy div, that is used to run the context when the page is accessed
        self.dummy_div = html.Div([], id="projection-page-dummy-div")

        # define the layout
        self.layout = html.Div(
            [
                self.display,
                self.main.store,
                html.Br(),
                dmc.LoadingOverlay(
                    self.projection_container,
                ),
                self.dummy_div,
            ],
            style={"margin": "2%"},
        )

        dash.register_page(
            "projection",
            layout=self.layout,
        )

        self.main.app.callback(
            Output("projection-page-display", "children"),
            Output("projection-page-plot", "figure"),
            Output("ngl-container", "children"),
            Input("projection-page-plot", "clickData"),
            Input("projection-page-plot", "selectedData"),
            Input("projection-page-plot", "relayoutData"),
            State("main-store", "data"),
        )(self.interact)

    @property
    def molStyles(self):
        molstyles_dict = {
            "representations": ["cartoon"],
            "chosenAtomsColor": "white",
            "chosenAtomsRadius": 1,
            "molSpacingXaxis": 100,
        }
        return molstyles_dict

    def interact(
        self, click_on_plot, select_in_plot, relayoutdata, main_store
    ) -> tuple[html.P, go.FigureWidget, Any]:
        """Interactive elements:

        * Click on Scatter Point
        * Buttons:
            * switch between cluster and scatter
            * cluster
            * generate
        * slider
        * dropdown
        * Progress

        """
        empty = [
            html.P("Go to the 'Load' page to load data."),
            self.figure_widget,
            [],
        ]
        triggered_id = ctx.triggered_id
        if triggered_id == "projection-page-plot":
            if click_on_plot is None:
                raise PreventUpdate
            index = [p["pointIndex"] for p in click_on_plot["points"]][0]
            frame = self.main.trajs.get_single_frame(index)
            marker = {
                "color": [
                    "#1f77b4" if i != index else "#ff7f0e"
                    for i in range(len(self.lowd))
                ],
                "size": [7 if i != index else 20 for i in range(len(self.lowd))],
            }
            self.figure_widget.update_traces(
                marker=marker, selector=({"name": "scatter"})
            )
            viewer = dashbio.NglMoleculeViewer(
                data=traj_to_pdb(frame.traj),
                molStyles=self.molStyles,
            )
            empty[2] = [viewer]
            print(f"{click_on_plot=} {empty[2]=}")
            return tuple(empty)

        if self.main.trajs is None:
            if main_store is not None:
                if main_store != {}:
                    self.main.traj_page.parse_trajs(main_store)
        if self.main.autoencoder is None:
            if main_store is not None:
                if main_store != {}:
                    if main_store["traj_type"] == "project":
                        empty[0] = html.P(
                            f"The project {main_store['traj']} has no autoencoder associated to it."
                        )
                    else:
                        empty[0] = html.P(f"The loaded trajs are not trained jet.")

        if self.main.autoencoder is not None:
            self.highd = self._highd_parser(
                self.main.autoencoder, highd=None, trajs=self.main.trajs
            )
            self.lowd = self._lowd_parser(
                self.main.autoencoder, lowd=None, trajs=self.main.trajs
            )
            empty[0] = html.P(f"Interact with the plot to view conformations.")

        # parse scatter
        if self.scatter is None and self.main.autoencoder is not None:
            self.lowd_dim = self.lowd.shape[1]
            self.scatter = go.Scattergl(
                x=self.lowd[:, 0],
                y=self.lowd[:, 1],
                mode="markers",
                name="scatter",
                marker={
                    "size": [10 for i in range(len(self.lowd))],
                    "color": ["#1f77b4" for i in range(len(self.lowd))],
                },
            )

            self.figure_widget.add_trace(self.scatter)

        print(
            f"Interacting {triggered_id=} {click_on_plot=} {select_in_plot=} {relayoutdata=} {self.main.trajs=} {main_store=} {self.scatter=}"
        )

        return tuple(empty)


class Dashboard:
    _encodermap_logo = "https://raw.githubusercontent.com/AG-Peter/encodermap/main/pic/logo_cube_300.png"

    def __init__(self):
        # create a dir for cache
        if self.local:
            self.cache_dir = Path("/tmp/encodermap_dash_cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise NotImplementedError("No cachedir in non-local mode.")

        # create the app and register the main page
        self.server = flask.Flask(__name__)
        self.store = dcc.Store(id="main-store", storage_type="session")
        self.trajs = None
        self.autoencoder = None
        self.app = Dash(
            server=self.server,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                dbc.icons.FONT_AWESOME,
            ],
            use_pages=True,
            pages_folder="",
        )
        vault_file = Path(__file__).resolve().parent.parent.parent / "act.vault"
        if vault_file.is_file():
            username = get_assign_from_file(vault_file, "DASH_USER")
            password = get_assign_from_file(vault_file, "DASH_PASSWORD")
            auth = dash_auth.BasicAuth(self.app, {username: password})
        self.app.title = "EncoderMap Dashboard"
        self.app._favicon = "favicon.ico"

        # theme changer
        self.theme_changer = ThemeChangerAIO(
            aio_id="theme-change",
            button_props={
                "color": "secondary",
                "class_name": "me-1",
                "outline": False,
                "style": {"margin-top": "5px"},
            },
            radio_props={
                "persistence": True,
                "persistence_type": "session",
            },
        )

        # create other pages
        if self.debug:
            self.debug_page = DebugPage()
        if self.local:
            self.upload_traj_page = LocalUploadTraj(self)
        else:
            raise Exception("Write a non-local upload page")
        self.traj_page = TrajPage(self)
        self.top_page = TopPage(self)
        self.projection_page = ProjectionPage(self)

        # collect instance attributes
        if self.local:
            self._greeting = f"EncoderMap Dashboard for {os.getlogin()}"
        else:
            self._greeting = "EncoderMap Dashboard"
        self._figures = []

        # init methods are divided into distinct methods to make them more legible
        dash.register_page("home", path="/", layout=self.layout)
        self.app.layout = self.app_layout

        # decorate the class callbacks
        callback(
            Output("placeholder", "children"),
            Input(ThemeChangerAIO.ids.radio("theme"), "value"),
            prevent_initial_call=True,
        )(self.update_theme)

    @property
    def app_layout(self):
        return html.Div(
            [
                self.navbar,
                dash.page_container,
                self.store,
                self.placeholder,
            ]
        )

    @property
    def layout(self):
        layout = html.Div(
            dbc.Container(
                [
                    html.H1("EncoderMap dashboard", className="display-3"),
                    html.P(
                        "Use the 'Upload' page to upload your MD data.",
                        className="lead",
                    ),
                    html.Hr(className="my-2"),
                    html.P(
                        [
                            "Check out EncoderMap's GitHub page: ",
                            html.A(
                                "https://github.com/AG-Peter/encodermap",
                                href="https://github.com/AG-Peter/encodermap",
                            ),
                        ]
                    ),
                    html.P(
                        [
                            "Read EncoderMap's documentation: ",
                            html.A(
                                "https://ag-peter.github.io/encodermap/",
                                href="https://ag-peter.github.io/encodermap/",
                            ),
                        ]
                    ),
                    html.P(
                        [
                            "Give credit to the authors: ",
                            html.Ul(
                                [
                                    html.Li(
                                        html.A(
                                            "https://pubs.acs.org/doi/abs/10.1021/acs.jctc.8b00975",
                                            href="https://pubs.acs.org/doi/abs/10.1021/acs.jctc.8b00975",
                                        ),
                                    ),
                                    html.Li(
                                        html.A(
                                            "https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00675",
                                            href="https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00675",
                                        ),
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                fluid=True,
                className="py-3",
            ),
            className="p-3 bg-light rounded-3",
        )
        return layout

    @property
    def navbar(self):
        navbar = dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        # Use row and col to control vertical alignment of logo / brand
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Img(src=self._encodermap_logo, height="30px")
                                ),
                                dbc.Col(
                                    dbc.NavbarBrand(self._greeting, className="ms-2")
                                ),
                                dbc.Col(
                                    [
                                        dbc.Nav(
                                            [
                                                dbc.NavItem(
                                                    dbc.NavLink(
                                                        page["name"],
                                                        href=page["relative_path"],
                                                    )
                                                )
                                                for page in dash.page_registry.values()
                                                if page["name"] != "Home"
                                            ],
                                            navbar=True,
                                        ),
                                    ],
                                    width={"size": "auto"},
                                ),
                            ],
                            align="center",
                            className="g-0",
                        ),
                        href="/",
                        style={"textDecoration": "none"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Nav(
                                        [
                                            dbc.NavItem(
                                                dbc.NavLink(
                                                    html.I(
                                                        className="fa-solid fa-heart",
                                                        style={"font-size": "1.5em"},
                                                    ),
                                                    href="https://www.chemie.uni-konstanz.de/ag-peter/",
                                                )
                                            ),
                                            dbc.NavItem(
                                                dbc.NavLink(
                                                    html.I(
                                                        className="fa-solid fa-book-open",
                                                        style={"font-size": "1.5em"},
                                                    ),
                                                    href="https://ag-peter.github.io/encodermap/",
                                                )
                                            ),
                                            dbc.NavItem(
                                                dbc.NavLink(
                                                    html.I(
                                                        className="fa-brands fa-square-github",
                                                        style={"font-size": "1.5em"},
                                                    ),
                                                    href="https://github.com/AG-Peter/encodermap",
                                                )
                                            ),
                                            dbc.NavItem(self.theme_changer),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        align="center",
                    ),
                ],
                fluid=True,
            ),
            color="primary",
            dark=True,
            style={"justify-content": "left"},
        )
        return navbar

    @property
    def placeholder(self):
        return html.P(id="placeholder")

    def update_theme(self, theme):
        for fig in self._figures:
            template = template_from_url(theme)
            fig.template = template

    @property
    def debug(self):
        return os.getenv("ENCODERMAP_DASH_DEBUG", "False") == "True"

    @property
    def local(self):
        return os.getenv("ENCODERMAP_DASH_RUN_LOCAL", "False") == "True"

    def run(self, *args, **kwargs):
        self.app.run(*args, host="0.0.0.0", debug=self.debug, **kwargs)


################################################################################
# Main
################################################################################


if __name__ == "__main__":
    dashboard = Dashboard()

    dashboard.run()
