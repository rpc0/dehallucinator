# -*- coding: utf-8 -*-
"""Public section, including homepage and signup."""
from flask import Blueprint, render_template

blueprint = Blueprint("public", __name__, static_folder="../static")


@blueprint.route("/", methods=["GET", "POST"])
def home():
    """Home page."""
    return render_template("public/home.html")


@blueprint.route("/about/")
def about():
    """About page."""
    return render_template("public/about.html")
