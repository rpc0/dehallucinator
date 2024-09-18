# -*- coding: utf-8 -*-
"""Public section, including homepage and signup."""
import deh.settings as settings
import requests
from flask import Blueprint, render_template, request

blueprint = Blueprint("public", __name__, static_folder="../static")


@blueprint.route("/", methods=["GET", "POST"])
def home():
    """Home page."""
    return render_template("public/home.html")


@blueprint.route("/about/")
def about():
    """About page."""
    return render_template("public/about.html")
