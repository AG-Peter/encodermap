# -*- coding: utf-8 -*-
# encodermap/plotting/dashboard.py
################################################################################
# Encodermap: A python library for dimensionality reduction.
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
import os
from pathlib import Path
from typing import TYPE_CHECKING

# Third Party Imports
import dash
import dash_auth
import dash_bootstrap_components as dbc
import flask
import lipsum
from dash import Dash, Input, Output, State, callback, ctx, dcc, html
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url

# Encodermap imports
import encodermap as em
from encodermap.kondata import get_assign_from_file


# from collections.abc import


################################################################################
# Globals
################################################################################


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
            [dbc.Card([], id="traj-upload-display")],
            fluid=True,
        )

        # the upload area
        self.upload_card_body = dbc.CardBody(
            [
                html.H4("Upload files"),
                dcc.Upload(
                    id="traj-upload-data-input",
                    children=html.Div(
                        ["Drag and Drop or ", html.A("Select Files")],
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
        self.upload_card_body_empty = dbc.CardBody(
            [
                html.H4("Files uploaded"),
                dbc.Placeholder(
                    button=True,
                    color="secondary",
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
                ),
            ],
        )
        self.upload_card = dbc.Col(
            [
                dbc.Card(
                    self.upload_card_body,
                    id="traj-upload-data-input-card",
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
                    id="traj-upload-paths-input",
                    class_name="h-50",
                ),
            ],
        )
        self.text_area_card_body_empty = dbc.CardBody(
            [
                html.H4("Files uploaded"),
                dbc.Placeholder(
                    button=True,
                    color="secondary",
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
                ),
            ],
        )
        self.text_area_card = dbc.Col(
            [
                dbc.Card(
                    self.text_area_card_body,
                    id="traj-upload-paths-card",
                    class_name="h-100",
                ),
            ],
            width=6,
            class_name="col-sm-6 col-lg-6",
        )

        # the project area
        self.input_card_body = dbc.CardBody(
            [
                html.H4("EncoderMap project"),
                dbc.Input(
                    placeholder="linear_dimers",
                    id="traj-upload-project-input",
                    class_name="h-50",
                ),
            ],
        )
        self.input_card_body_empty = dbc.CardBody(
            [
                html.H4("Files uploaded"),
                dbc.Placeholder(
                    button=True,
                    color="secondary",
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
                ),
            ],
        )
        self.input_card = dbc.Col(
            [
                dbc.Card(
                    self.input_card_body,
                    id="traj-upload-project-card",
                    class_name="h-100",
                ),
            ],
            width=3,
            class_name="col-sm-6 col-lg-3",
        )

        # the complete container
        self.upload_container = dbc.Container(
            [
                dbc.Row(
                    [
                        self.upload_card,
                        self.text_area_card,
                        self.input_card,
                    ],
                    class_name="h-50",
                ),
                dbc.Row(
                    [
                        dbc.Button(
                            "Upload",
                            id="traj-upload-button",
                            style={"width": "100%"},
                        )
                    ]
                ),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Button(
                            "Reset",
                            id="traj-upload-reset-button",
                            style={"width": "100%"},
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
                self.upload_container,
            ],
        )

        # define local callbacks
        self.main.app.callback(
            Output("traj-upload-display", "children"),
            Output("traj-upload-data-input-card", "children"),
            Output("traj-upload-paths-card", "children"),
            Output("traj-upload-project-card", "children"),
            Output("main-store", "data"),
            Input("traj-upload-button", "n_clicks"),
            Input("traj-upload-reset-button", "n_clicks"),
            State("main-store", "data"),
            State("traj-upload-data-input", "contents"),
            State("traj-upload-data-input", "filename"),
            State("traj-upload-data-input", "last_modified"),
            State("traj-upload-paths-input", "value"),
            State("traj-upload-project-input", "value"),
        )(self.load_trajs)

        # register the pae
        dash.register_page("upload", layout=self.layout)

    def load_trajs(
        self,
        upload_n_clicks,  # Input
        reset_n_clicks,  # Input
        main_store,  # State
        list_of_contents,  # State
        list_of_names,  # State
        list_of_dates,  # State
        textarea_value,  # State
        project_value,  # State
    ):
        triggered_id = ctx.triggered_id

        # reset button pressed
        if triggered_id == "traj-upload-reset-button":
            if hasattr(self.main, "trajs"):
                del self.main.trajs
            return (
                dbc.CardBody(f"Session was reset. Choose MD data to upload."),
                self.upload_card_body,
                self.text_area_card_body,
                self.input_card_body,
                {},
            )

        empty = [
            dbc.CardBody(f"Data already loaded. Press Reset to reset the session."),
            self.upload_card_body_empty,
            self.text_area_card_body_empty,
            self.input_card_body_empty,
            main_store,
        ]

        if triggered_id is None:
            if isinstance(main_store, dict):
                if "traj_type" in main_store:
                    return tuple(empty)
            return (
                dbc.CardBody(f"Choose MD data to upload."),
                self.upload_card_body,
                self.text_area_card_body,
                self.input_card_body,
                main_store,
            )

        # upload button pressed
        uploaded_any = (
            list_of_contents is not None
            or textarea_value is not None
            or project_value is not None
        )
        if triggered_id == "traj-upload-button" and not uploaded_any:
            return (
                dbc.CardBody(
                    f"Place files in the upload window, or enter local files, "
                    f"or a project name before pressing upload."
                ),
                self.upload_card_body,
                self.text_area_card_body,
                self.input_card_body,
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
                f"Loading trajs. Go to the 'Traj' page to look at your data."
            )
        elif project_value is not None:
            main_store.update({"traj_type": "project", "traj": project_value})
            self.main.traj_page.parse_trajs(main_store)
            empty[0] = dbc.CardBody(
                f"Loading project '{project_value}'. Go to the 'Traj' page to look at your data."
            )

        empty[-1] = main_store
        return tuple(empty)


class TrajPage:
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
        return html.P(f"{args=} {kwargs=}")

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
                    html.Div(html.P("Current selected num"), id="traj-page-display"),
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
        else:
            pass

        return layout

    def parse_trajs(self, data):
        if data["traj_type"] == "project":
            self.main.trajs = em.load_project(data["traj"])
        else:
            raise NotImplementedError


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
        self.app.run(*args, debug=self.debug, **kwargs)


################################################################################
# Main
################################################################################


if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run()
