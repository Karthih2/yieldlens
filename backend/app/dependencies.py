from fastapi import Request, Depends
from sqlalchemy.orm import Session
from app.db import get_db

def get_model(request: Request):
    return request.app.state.model

def get_explainer(request: Request):
    return request.app.state.explainer

def get_shortlist(request: Request):
    return request.app.state.shortlist_cols

def get_medians(request: Request):
    return request.app.state.medians

def get_baseline_distribution(request: Request):
    return request.app.state.baseline_distribution

def get_stability_df(request: Request):
    return request.app.state.stability_df
