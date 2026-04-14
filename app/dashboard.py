"""
dashboard.py
────────────
Cinemate — Plotly Dash Analytics Dashboard

6 tabs covering:
    1. Dataset Overview
    2. Model Performance
    3. Recommendation Analysis
    4. A/B Test Results
    5. Business Impact
    6. Bias & Fairness

Run:
    python app/dashboard.py
    Visit: http://localhost:8050
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ── Paths ──────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.dirname(__file__))
DATA_DIR           = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR         = os.path.join(BASE_DIR, "models")


# ══════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════

def load_all_data():
    """Load all JSON results and parquet files."""
    data = {}

    # Model results
    for fname, key in [
        ("two_tower_results.json",    "tt"),
        ("ncf_results.json",          "ncf"),
        ("svd_baseline_results.json", "svd"),
        ("ab_test_results.json",      "ab"),
        ("business_impact.json",      "biz"),
        ("bias_results.json",         "bias"),
        ("rec_analysis.json",         "rec"),
    ]:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            with open(path, "r") as f:
                data[key] = json.load(f)
        else:
            data[key] = {}

    # Dataset
    data['train'] = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "train.parquet")
    )
    data['movies'] = pd.read_parquet(
        os.path.join(PROCESSED_DATA_DIR, "movies_clean.parquet")
    )

    with open(os.path.join(
        PROCESSED_DATA_DIR, "dataset_constants.pkl"
    ), "rb") as f:
        data['constants'] = pickle.load(f)

    return data


DATA = load_all_data()


# ══════════════════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════════════════

COLORS = {
    'bg'       : '#0A0A0F',
    'surface'  : '#13131A',
    'border'   : '#2A2A3A',
    'primary'  : '#6366F1',
    'success'  : '#10B981',
    'warning'  : '#F59E0B',
    'danger'   : '#EF4444',
    'text'     : '#F1F5F9',
    'muted'    : '#64748B',
    'random'   : '#94A3B8',
    'pop'      : '#64748B',
    'svd'      : '#3B82F6',
    'ncf'      : '#8B5CF6',
    'tt'       : '#10B981',
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor = COLORS['bg'],
        plot_bgcolor  = COLORS['surface'],
        font          = dict(color=COLORS['text'],
                             family='system-ui'),
        title         = dict(font=dict(size=16,
                             color=COLORS['text'])),
        xaxis         = dict(
            gridcolor    = COLORS['border'],
            linecolor    = COLORS['border'],
            showgrid     = True,
        ),
        yaxis         = dict(
            gridcolor    = COLORS['border'],
            linecolor    = COLORS['border'],
            showgrid     = True,
        ),
        legend        = dict(
            bgcolor      = COLORS['surface'],
            bordercolor  = COLORS['border'],
            borderwidth  = 1,
        ),
        margin        = dict(l=40, r=20, t=50, b=40),
    )
)


# ══════════════════════════════════════════════════════════
# HELPER COMPONENTS
# ══════════════════════════════════════════════════════════

def kpi_card(value, label, color=None):
    """Single KPI metric card."""
    color = color or COLORS['primary']
    return html.Div([
        html.Div(str(value),
                 style={'fontSize':'32px', 'fontWeight':'700',
                        'color': color}),
        html.Div(label,
                 style={'fontSize':'12px', 'color': COLORS['muted'],
                        'marginTop':'4px'}),
    ], style={
        'background'   : COLORS['surface'],
        'border'       : f"1px solid {COLORS['border']}",
        'borderRadius' : '12px',
        'padding'      : '20px',
        'textAlign'    : 'center',
        'flex'         : '1',
    })


def section_header(title, subtitle=""):
    return html.Div([
        html.H3(title, style={
            'color'      : COLORS['text'],
            'margin'     : '0 0 4px 0',
            'fontSize'   : '18px',
            'fontWeight' : '600',
        }),
        html.P(subtitle, style={
            'color'    : COLORS['muted'],
            'margin'   : '0',
            'fontSize' : '13px',
        }) if subtitle else html.Div(),
        html.Hr(style={
            'border'     : f'1px solid {COLORS["border"]}',
            'margin'     : '12px 0',
        })
    ])


def apply_theme(fig):
    """Apply dark theme to any plotly figure."""
    fig.update_layout(
        **PLOTLY_TEMPLATE['layout']
    )
    return fig


# ══════════════════════════════════════════════════════════
# TAB CONTENT BUILDERS
# ══════════════════════════════════════════════════════════

# ── Tab 1 — Dataset Overview ───────────────────────────────

def build_tab_dataset():
    train     = DATA['train']
    constants = DATA['constants']
    movies    = DATA['movies']

    NUM_USERS  = constants['NUM_USERS']
    NUM_MOVIES = constants['NUM_MOVIES']
    N_RATINGS  = len(train)
    sparsity   = 1 - N_RATINGS / (NUM_USERS * NUM_MOVIES)

    # Rating distribution
    rating_counts = train['rating'].astype(float).value_counts(
    ).sort_index().reset_index()
    rating_counts.columns = ['rating', 'count']
    rating_counts['pct']  = (
        rating_counts['count'] / len(train) * 100
    ).round(1)

    fig_rating = go.Figure()
    fig_rating.add_trace(go.Bar(
        x     = rating_counts['rating'].astype(str),
        y     = rating_counts['count'],
        text  = rating_counts['pct'].apply(lambda x: f'{x}%'),
        textposition = 'outside',
        marker_color = COLORS['primary'],
    ))
    fig_rating.update_layout(
        **PLOTLY_TEMPLATE['layout']
    )
    # Update the text specifically after applying the template
    fig_rating.update_layout(
        title_text='Rating Distribution',
        xaxis_title='Rating',
        yaxis_title='Count'
)

    # Genre distribution
    movies['genre_list'] = movies['genres_clean'].str.split()
    genre_counts         = {}
    for genres in movies['genre_list'].dropna():
        for g in genres:
            genre_counts[g] = genre_counts.get(g, 0) + 1
    genre_df = pd.DataFrame(
        genre_counts.items(), columns=['genre', 'count']
    ).sort_values('count', ascending=True).tail(15)

    fig_genre = go.Figure(go.Bar(
        x            = genre_df['count'],
        y            = genre_df['genre'],
        orientation  = 'h',
        marker_color = COLORS['primary'],
    ))
    fig_genre.update_layout(
        title       = 'Movie Count by Genre',
        xaxis_title = 'Number of Movies',
        **PLOTLY_TEMPLATE['layout']
    )

    # Ratings per user distribution
    ratings_per_user = train.groupby('user_idx').size()
    fig_user = go.Figure(go.Histogram(
        x           = ratings_per_user.values,
        nbinsx      = 80,
        marker_color= COLORS['primary'],
    ))
    fig_user.update_layout(
        title       = 'Ratings per User Distribution (log y)',
        xaxis_title = 'Number of Ratings',
        yaxis_title = 'Number of Users',
        yaxis_type  = 'log',
        **PLOTLY_TEMPLATE['layout']
    )

    return html.Div([
        section_header(
            "Dataset Overview",
            "MovieLens 33M — key statistics and distributions"
        ),

        # KPI row
        html.Div([
            kpi_card(f"{NUM_USERS:,}",  "Total Users"),
            kpi_card(f"{NUM_MOVIES:,}", "Total Movies"),
            kpi_card(f"{N_RATINGS/1e6:.1f}M", "Total Ratings"),
            kpi_card(f"{sparsity:.2%}", "Matrix Sparsity",
                     COLORS['warning']),
            kpi_card("33M",  "Dataset Version"),
            kpi_card("80/20","Train/Test Split"),
        ], style={'display':'flex', 'gap':'12px',
                  'marginBottom':'24px'}),

        # Charts row 1
        html.Div([
            dcc.Graph(figure=fig_rating,
                      style={'flex':'1'}),
            dcc.Graph(figure=fig_genre,
                      style={'flex':'1'}),
        ], style={'display':'flex', 'gap':'16px',
                  'marginBottom':'16px'}),

        # Charts row 2
        html.Div([
            dcc.Graph(figure=fig_user,
                      style={'flex':'1'}),
            html.Div([
                section_header("Key EDA Findings"),
                html.Ul([
                    html.Li(
                        "99.88% sparse user-item matrix — "
                        "typical for real-world recommenders",
                        style={'color': COLORS['text'],
                               'marginBottom':'8px'}
                    ),
                    html.Li(
                        "Mean rating 3.54 (positive bias) → "
                        "BPR loss chosen over MSE",
                        style={'color': COLORS['text'],
                               'marginBottom':'8px'}
                    ),
                    html.Li(
                        "Power law distribution — top 10% movies "
                        "hold 80%+ of ratings",
                        style={'color': COLORS['text'],
                               'marginBottom':'8px'}
                    ),
                    html.Li(
                        "Time-based 80/20 split to prevent "
                        "timestamp leakage",
                        style={'color': COLORS['text'],
                               'marginBottom':'8px'}
                    ),
                    html.Li(
                        "38.2% users have < 20 ratings → "
                        "filtered to reduce noise",
                        style={'color': COLORS['text'],
                               'marginBottom':'8px'}
                    ),
                ], style={'paddingLeft':'20px'})
            ], style={
                'flex'         : '1',
                'background'   : COLORS['surface'],
                'border'       : f"1px solid {COLORS['border']}",
                'borderRadius' : '12px',
                'padding'      : '20px',
            }),
        ], style={'display':'flex', 'gap':'16px'}),
    ])


# ── Tab 2 — Model Performance ──────────────────────────────

def build_tab_models():
    tt  = DATA.get('tt',  {})
    ncf = DATA.get('ncf', {})
    svd = DATA.get('svd', {})

    # Metric comparison
    models    = ['Random', 'Popularity', 'SVD',
                 'NCF', 'Two-Tower']
    ndcg_vals = [
        svd.get('metrics',{}).get('random',{}).get('NDCG@10', 0),
        svd.get('metrics',{}).get('popularity',{}).get('NDCG@10', 0),
        svd.get('metrics',{}).get('svd',{}).get('NDCG@10', 0),
        ncf.get('metrics',{}).get('NDCG@10', 0),
        tt.get('metrics',{}).get('two_tower',{}).get('NDCG@10', 0),
    ]
    prec_vals = [
        svd.get('metrics',{}).get('random',{}).get('Precision@10', 0),
        svd.get('metrics',{}).get('popularity',{}).get('Precision@10', 0),
        svd.get('metrics',{}).get('svd',{}).get('Precision@10', 0),
        ncf.get('metrics',{}).get('Precision@10', 0),
        tt.get('metrics',{}).get('two_tower',{}).get('Precision@10', 0),
    ]
    rec_vals = [
        svd.get('metrics',{}).get('random',{}).get('Recall@10', 0),
        svd.get('metrics',{}).get('popularity',{}).get('Recall@10', 0),
        svd.get('metrics',{}).get('svd',{}).get('Recall@10', 0),
        ncf.get('metrics',{}).get('Recall@10', 0),
        tt.get('metrics',{}).get('two_tower',{}).get('Recall@10', 0),
    ]
    bar_colors = [
        COLORS['random'], COLORS['pop'], COLORS['svd'],
        COLORS['ncf'],    COLORS['tt']
    ]

    fig_ndcg = go.Figure()
    fig_ndcg.add_trace(go.Bar(
        x             = models,
        y             = ndcg_vals,
        marker_color  = bar_colors,
        text          = [f'{v:.4f}' for v in ndcg_vals],
        textposition  = 'outside',
    ))
    fig_ndcg.update_layout(
        title       = 'NDCG@10 — All Models',
        yaxis_title = 'NDCG@10',
        showlegend  = False,
        **PLOTLY_TEMPLATE['layout']
    )

    # Grouped bar — all metrics
    fig_all = go.Figure()
    for metric, vals, color in [
        ('NDCG@10',      ndcg_vals, COLORS['primary']),
        ('Precision@10', prec_vals, COLORS['success']),
        ('Recall@10',    rec_vals,  COLORS['warning']),
    ]:
        fig_all.add_trace(go.Bar(
            name         = metric,
            x            = models,
            y            = vals,
            marker_color = color,
        ))
    fig_all.update_layout(
        title       = 'All Metrics Comparison',
        yaxis_title = 'Score',
        barmode     = 'group',
        **PLOTLY_TEMPLATE['layout']
    )

    # Training loss curves
    ncf_history = ncf.get('training', {}).get('history', {})
    tt_history  = tt.get('training',  {}).get('history', {})

    fig_loss = go.Figure()
    if ncf_history.get('train_loss'):
        fig_loss.add_trace(go.Scatter(
            y      = ncf_history['train_loss'],
            x      = list(range(1, len(
                ncf_history['train_loss'])+1
            )),
            name   = 'NCF',
            line   = dict(color=COLORS['ncf'], width=2),
            mode   = 'lines+markers',
        ))
    if tt_history.get('train_loss'):
        fig_loss.add_trace(go.Scatter(
            y      = tt_history['train_loss'],
            x      = list(range(1, len(
                tt_history['train_loss'])+1
            )),
            name   = 'Two-Tower',
            line   = dict(color=COLORS['tt'], width=2),
            mode   = 'lines+markers',
        ))
    fig_loss.update_layout(
        title       = 'Training Loss (BPR) — NCF vs Two-Tower',
        xaxis_title = 'Epoch',
        yaxis_title = 'BPR Loss',
        **PLOTLY_TEMPLATE['layout']
    )

    # Model comparison table
    table_data = pd.DataFrame({
        'Model'        : models,
        'NDCG@10'      : [f'{v:.4f}' for v in ndcg_vals],
        'Precision@10' : [f'{v:.4f}' for v in prec_vals],
        'Recall@10'    : [f'{v:.4f}' for v in rec_vals],
        'vs SVD'       : [
            '—',
            f"{100*(ndcg_vals[1]-ndcg_vals[2])/max(ndcg_vals[2],1e-6):+.1f}%",
            'baseline',
            f"{100*(ndcg_vals[3]-ndcg_vals[2])/max(ndcg_vals[2],1e-6):+.1f}%",
            f"{100*(ndcg_vals[4]-ndcg_vals[2])/max(ndcg_vals[2],1e-6):+.1f}%",
        ]
    })

    fig_table = go.Figure(go.Table(
        header = dict(
            values    = list(table_data.columns),
            fill_color= COLORS['border'],
            font      = dict(color=COLORS['text'], size=13),
            align     = 'left',
            height    = 36,
        ),
        cells  = dict(
            values    = [table_data[c] for c in table_data.columns],
            fill_color= [[
                COLORS['surface'] if i % 2 == 0
                else '#16161F'
                for i in range(len(table_data))
            ]],
            font      = dict(color=COLORS['text'], size=12),
            align     = 'left',
            height    = 32,
        )
    ))
    fig_table.update_layout(
        title  = 'Complete Results Table',
        height = 280,
        **PLOTLY_TEMPLATE['layout']
    )

    return html.Div([
        section_header(
            "Model Performance",
            "Evaluation on same 2000 users | time-based split | NDCG@10 is primary metric"
        ),

        dcc.Graph(figure=fig_table),

        html.Div([
            dcc.Graph(figure=fig_ndcg, style={'flex':'1'}),
            dcc.Graph(figure=fig_all,  style={'flex':'1'}),
        ], style={'display':'flex', 'gap':'16px',
                  'marginBottom':'16px'}),

        dcc.Graph(figure=fig_loss),
    ])


# ── Tab 3 — Recommendation Analysis ───────────────────────

def build_tab_rec_analysis():
    rec = DATA.get('rec', {})

    coverage       = rec.get('catalogue_coverage', 0)
    personalisation= rec.get('personalisation_score', 0)
    avg_genres     = rec.get('avg_genres_per_user', 0)
    never_rec      = rec.get('never_recommended', 0)
    unique_rec     = rec.get('unique_movies_rec', 0)
    NUM_MOVIES     = DATA['constants']['NUM_MOVIES']

    # Coverage donut
    fig_cov = go.Figure(go.Pie(
        values = [unique_rec, never_rec],
        labels = ['Recommended', 'Never Recommended'],
        hole   = 0.6,
        marker = dict(colors=[
            COLORS['primary'], COLORS['border']
        ]),
        textinfo   = 'label+percent',
        textfont   = dict(color=COLORS['text']),
    ))
    fig_cov.update_layout(
        title       = f'Catalogue Coverage: {coverage:.1%}',
        annotations = [dict(
            text      = f'{coverage:.0%}',
            x=0.5, y=0.5,
            font_size = 24,
            showarrow = False,
            font      = dict(color=COLORS['primary'])
        )],
        **PLOTLY_TEMPLATE['layout']
    )

    # Personalisation gauge
    fig_pers = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = personalisation * 100,
        title = {'text': "Personalisation Score (%)"},
        gauge = {
            'axis' : {'range': [0, 100]},
            'bar'  : {'color': COLORS['primary']},
            'steps': [
                {'range': [0, 30],  'color': COLORS['danger']},
                {'range': [30, 60], 'color': COLORS['warning']},
                {'range': [60, 100],'color': COLORS['success']},
            ],
            'threshold': {
                'line' : {'color': 'white', 'width': 2},
                'value': personalisation * 100
            }
        },
        number = {'suffix': '%', 'font': {'size': 32}}
    ))
    fig_pers.update_layout(
        height = 300,
        **PLOTLY_TEMPLATE['layout']
    )

    # Genre coverage bar
    movies = DATA['movies']
    movies['genre_list'] = movies['genres_clean'].str.split()
    all_genres = {}
    for genres in movies['genre_list'].dropna():
        for g in genres:
            all_genres[g] = all_genres.get(g, 0) + 1

    genre_names = sorted(all_genres.keys())
    genre_total = [all_genres[g] for g in genre_names]

    fig_genre_cov = go.Figure()
    fig_genre_cov.add_trace(go.Bar(
        x            = genre_names,
        y            = genre_total,
        name         = 'In Catalogue',
        marker_color = COLORS['border'],
    ))
    fig_genre_cov.update_layout(
        title       = 'Genre Coverage in Catalogue',
        xaxis_title = 'Genre',
        yaxis_title = 'Movie Count',
        xaxis_tickangle = -45,
        **PLOTLY_TEMPLATE['layout']
    )

    return html.Div([
        section_header(
            "Recommendation Analysis",
            f"Based on {rec.get('n_sample_users', 500)} "
            f"sample users | top-{rec.get('top_k', 10)} recommendations"
        ),

        # KPI row
        html.Div([
            kpi_card(f"{coverage:.1%}", "Catalogue Coverage"),
            kpi_card(
                f"{personalisation:.3f}",
                "Personalisation Score",
                COLORS['success'] if personalisation > 0.5
                else COLORS['warning']
            ),
            kpi_card(f"{avg_genres:.1f}", "Avg Genres/User"),
            kpi_card(f"{unique_rec:,}", "Unique Movies Recommended"),
            kpi_card(f"{never_rec:,}", "Never Recommended",
                     COLORS['warning']),
        ], style={'display':'flex', 'gap':'12px',
                  'marginBottom':'24px'}),

        html.Div([
            dcc.Graph(figure=fig_cov,  style={'flex':'1'}),
            dcc.Graph(figure=fig_pers, style={'flex':'1'}),
        ], style={'display':'flex', 'gap':'16px',
                  'marginBottom':'16px'}),

        dcc.Graph(figure=fig_genre_cov),

        # Insights
        html.Div([
            section_header("Interpretation"),
            html.Div([
                html.Div([
                    html.Strong("Catalogue Coverage: ",
                                style={'color':COLORS['primary']}),
                    f"{coverage:.1%} of movies recommended across "
                    f"{rec.get('n_sample_users',500)} users. "
                    f"Lower coverage indicates popularity bias.",
                ], style={'color':COLORS['text'],
                          'marginBottom':'12px'}),
                html.Div([
                    html.Strong("Personalisation: ",
                                style={'color':COLORS['primary']}),
                    f"Score of {personalisation:.3f} means users "
                    f"receive meaningfully different recommendations "
                    f"(1 = completely unique, 0 = all identical).",
                ], style={'color':COLORS['text'],
                          'marginBottom':'12px'}),
            ])
        ], style={
            'background'  : COLORS['surface'],
            'border'      : f"1px solid {COLORS['border']}",
            'borderRadius': '12px',
            'padding'     : '20px',
            'marginTop'   : '16px',
        })
    ])


# ── Tab 4 — A/B Test ──────────────────────────────────────

def build_tab_ab():
    ab = DATA.get('ab', {})

    ctrl_mean  = ab.get('control_ndcg',   {}).get('mean', 0)
    treat_mean = ab.get('treatment_ndcg', {}).get('mean', 0)
    ctrl_std   = ab.get('control_ndcg',   {}).get('std',  0.05)
    treat_std  = ab.get('treatment_ndcg', {}).get('std',  0.18)

    lift_abs   = ab.get('lift', {}).get('absolute',    0)
    lift_rel   = ab.get('lift', {}).get('relative_pct',0)
    ci_low     = ab.get('lift', {}).get('ci_95_low',   0)
    ci_high    = ab.get('lift', {}).get('ci_95_high',  0)

    p_value    = ab.get('hypothesis_test', {}).get('p_value',  1)
    cohens_d   = ab.get('hypothesis_test', {}).get('cohens_d', 0)
    significant= ab.get('hypothesis_test', {}).get('significant', False)
    decision   = ab.get('decision', 'UNKNOWN')

    n_users    = ab.get('n_users', 2000)

    # Distribution comparison (simulated from stats)
    x = np.linspace(-0.1, 1.1, 300)
    from scipy.stats import norm
    ctrl_dist  = norm.pdf(x, ctrl_mean,  ctrl_std)
    treat_dist = norm.pdf(x, treat_mean, treat_std)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Scatter(
        x    = x, y = ctrl_dist,
        name = f'Control (Random) μ={ctrl_mean:.4f}',
        fill = 'tozeroy',
        line = dict(color=COLORS['danger'], width=2),
        fillcolor='rgba(239,68,68,0.15)',
    ))
    fig_dist.add_trace(go.Scatter(
        x    = x, y = treat_dist,
        name = f'Treatment (Two-Tower) μ={treat_mean:.4f}',
        fill = 'tozeroy',
        line = dict(color=COLORS['success'], width=2),
        fillcolor='rgba(16,185,129,0.15)',
    ))
    fig_dist.add_vline(
        x=ctrl_mean, line_dash="dash",
        line_color=COLORS['danger'], opacity=0.7
    )
    fig_dist.add_vline(
        x=treat_mean, line_dash="dash",
        line_color=COLORS['success'], opacity=0.7
    )
    fig_dist.update_layout(
        title       = 'NDCG@10 Distribution — Control vs Treatment',
        xaxis_title = 'NDCG@10',
        yaxis_title = 'Density',
        **PLOTLY_TEMPLATE['layout']
    )

    # Lift with CI
    fig_lift = go.Figure()
    fig_lift.add_trace(go.Bar(
        x            = ['Two-Tower vs Random'],
        y            = [lift_rel],
        error_y      = dict(
            type      = 'data',
            symmetric = False,
            array     = [
                abs(100*ci_high/max(ctrl_mean,1e-6) - lift_rel)
            ],
            arrayminus= [
                abs(lift_rel - 100*ci_low/max(ctrl_mean,1e-6))
            ],
        ),
        marker_color  = COLORS['success'],
        text          = [f'+{lift_rel:.1f}%'],
        textposition  = 'outside',
    ))
    fig_lift.add_hline(
        y=0, line_dash='dash',
        line_color=COLORS['muted']
    )
    fig_lift.update_layout(
        title       = f'Relative NDCG Lift with 95% CI',
        yaxis_title = 'Relative Lift (%)',
        **PLOTLY_TEMPLATE['layout']
    )

    # Stats summary table
    stats_table = go.Figure(go.Table(
        header = dict(
            values    = ['Metric', 'Value', 'Interpretation'],
            fill_color= COLORS['border'],
            font      = dict(color=COLORS['text'], size=13),
            align     = 'left',
            height    = 36,
        ),
        cells  = dict(
            values = [
                ['Control NDCG@10', 'Treatment NDCG@10',
                 'Absolute Lift', 'Relative Lift',
                 '95% CI', 'p-value',
                 "Cohen's d", 'N per group', 'Decision'],
                [f'{ctrl_mean:.4f}', f'{treat_mean:.4f}',
                 f'+{lift_abs:.4f}', f'+{lift_rel:.1f}%',
                 f'[{ci_low:.4f}, {ci_high:.4f}]',
                 f'{p_value:.6f}',
                 f'{cohens_d:.4f}', f'{n_users:,}',
                 decision],
                ['Random baseline', 'Our model',
                 'Raw NDCG improvement',
                 'Percentage improvement',
                 'True lift range (95% confidence)',
                 '< 0.05 = significant',
                 'Effect size magnitude',
                 'Users in experiment',
                 '🚀 SHIP' if decision=='SHIP'
                 else '⚠️ DO NOT SHIP'],
            ],
            fill_color = [[
                COLORS['surface'] if i % 2 == 0
                else '#16161F'
                for i in range(9)
            ]],
            font   = dict(color=[
                [COLORS['text']] * 9,
                [COLORS['text']] * 9,
                [COLORS['success'] if i == 8
                 and decision == 'SHIP'
                 else COLORS['muted']
                 for i in range(9)],
            ], size=12),
            align  = 'left',
            height = 32,
        )
    ))
    stats_table.update_layout(
        title  = 'Statistical Test Results',
        height = 380,
        **PLOTLY_TEMPLATE['layout']
    )

    # Ship/Don't Ship card
    ship_color  = COLORS['success'] if decision == 'SHIP' \
                  else COLORS['danger']
    ship_icon   = '🚀' if decision == 'SHIP' else '⚠️'
    ship_msg    = (
        "All criteria met. Two-Tower recommender shows "
        "statistically significant improvement. "
        "Recommended for production deployment."
        if decision == 'SHIP'
        else "Not all criteria met. Review results before deploying."
    )

    decision_card = html.Div([
        html.H2(f"{ship_icon} {decision}",
                style={'color': ship_color,
                       'margin': '0 0 8px 0',
                       'fontSize': '28px'}),
        html.P(ship_msg,
               style={'color': COLORS['text'],
                      'margin': '0', 'fontSize': '14px'}),
        html.Div([
            html.Span(f"p={p_value:.4f}",
                      style={'background': '#1E2A1E',
                             'color': COLORS['success'],
                             'padding': '4px 12px',
                             'borderRadius': '6px',
                             'marginRight': '8px',
                             'fontSize': '13px'}),
            html.Span(f"d={cohens_d:.3f}",
                      style={'background': '#1A1E2A',
                             'color': COLORS['primary'],
                             'padding': '4px 12px',
                             'borderRadius': '6px',
                             'marginRight': '8px',
                             'fontSize': '13px'}),
            html.Span(f"+{lift_rel:.1f}% lift",
                      style={'background': '#1E2A1E',
                             'color': COLORS['success'],
                             'padding': '4px 12px',
                             'borderRadius': '6px',
                             'fontSize': '13px'}),
        ], style={'marginTop': '12px'}),
    ], style={
        'background'   : COLORS['surface'],
        'border'       : f'2px solid {ship_color}',
        'borderRadius' : '12px',
        'padding'      : '24px',
        'marginBottom' : '24px',
    })

    return html.Div([
        section_header(
            "A/B Test Results",
            f"Control: random recommendations | "
            f"Treatment: Two-Tower | N={n_users:,} users per group"
        ),
        decision_card,
        html.Div([
            dcc.Graph(figure=fig_dist, style={'flex':'1'}),
            dcc.Graph(figure=fig_lift, style={'flex':'1'}),
        ], style={'display':'flex', 'gap':'16px',
                  'marginBottom':'16px'}),
        dcc.Graph(figure=stats_table),
    ])


# ── Tab 5 — Business Impact ────────────────────────────────

def build_tab_business():
    biz = DATA.get('biz', {})
    ab  = DATA.get('ab',  {})

    monthly_lift = biz.get('monthly_lift_inr', 0)
    annual_lift  = biz.get('annual_lift_inr',  0)
    net_annual   = biz.get('net_annual_inr',   0)
    roi          = biz.get('roi_pct',           0)
    payback      = biz.get('payback_months',    0)
    assumptions  = biz.get('assumptions',       {})

    MAU  = assumptions.get('monthly_active_users', 100000)
    ARPU = assumptions.get('avg_revenue_per_user', 199)
    ret_lift = assumptions.get('rec_retention_lift', 0.05)

    # Sensitivity heatmap
    ret_lifts  = [0.02, 0.03, 0.05, 0.07, 0.10]
    arpu_vals  = [99, 149, 199, 299, 499]
    matrix     = np.array([
        [MAU * r * a * 12 / 100000
         for a in arpu_vals]
        for r in ret_lifts
    ])

    fig_heat = go.Figure(go.Heatmap(
        z          = matrix,
        x          = [f'₹{a}' for a in arpu_vals],
        y          = [f'{r*100:.0f}%' for r in ret_lifts],
        colorscale = 'YlGn',
        text       = [[f'₹{v:.0f}L' for v in row]
                      for row in matrix],
        texttemplate = '%{text}',
        textfont     = dict(size=11),
    ))
    fig_heat.update_layout(
        title       = 'Annual Revenue Lift (₹ Lakhs) — Sensitivity',
        xaxis_title = 'Average Revenue Per User (₹/month)',
        yaxis_title = 'Retention Lift from Recommendations',
        **PLOTLY_TEMPLATE['layout']
    )

    # ROI waterfall
    dev_cost    = assumptions.get('dev_cost_inr', 500000)
    infra_annual= assumptions.get(
        'infra_cost_monthly', 10000
    ) * 12

    fig_waterfall = go.Figure(go.Waterfall(
        name      = "ROI",
        orientation = "v",
        measure   = ["relative", "relative",
                     "relative", "total"],
        x         = ["Annual Revenue Lift",
                     "Dev Cost",
                     "Infrastructure",
                     "Net Annual Value"],
        y         = [annual_lift/100000,
                     -dev_cost/100000,
                     -infra_annual/100000,
                     0],
        text      = [
            f'+₹{annual_lift/100000:.1f}L',
            f'-₹{dev_cost/100000:.1f}L',
            f'-₹{infra_annual/100000:.1f}L',
            f'₹{net_annual/100000:.1f}L',
        ],
        textposition = "outside",
        connector    = dict(line=dict(
            color=COLORS['muted'], width=1
        )),
        increasing   = dict(marker=dict(
            color=COLORS['success']
        )),
        decreasing   = dict(marker=dict(
            color=COLORS['danger']
        )),
        totals       = dict(marker=dict(
            color=COLORS['primary']
        )),
    ))
    fig_waterfall.update_layout(
        title       = 'Annual ROI Breakdown (₹ Lakhs)',
        yaxis_title = '₹ Lakhs',
        showlegend  = False,
        **PLOTLY_TEMPLATE['layout']
    )

    return html.Div([
        section_header(
            "Business Impact",
            "Conservative estimates based on 100K MAU, "
            "5% retention lift, ₹199 ARPU"
        ),

        # KPI row
        html.Div([
            kpi_card(
                f"₹{monthly_lift/100000:.1f}L",
                "Monthly Revenue Lift",
                COLORS['success']
            ),
            kpi_card(
                f"₹{annual_lift/100000:.1f}L",
                "Annual Revenue Lift",
                COLORS['success']
            ),
            kpi_card(
                f"₹{net_annual/100000:.1f}L",
                "Net Annual Value",
                COLORS['primary']
            ),
            kpi_card(
                f"{roi:.0f}%",
                "ROI",
                COLORS['success'] if roi > 0
                else COLORS['danger']
            ),
            kpi_card(
                f"{payback:.1f} mo",
                "Payback Period",
                COLORS['warning']
            ),
        ], style={'display':'flex', 'gap':'12px',
                  'marginBottom':'24px'}),

        html.Div([
            dcc.Graph(figure=fig_waterfall,
                      style={'flex':'1'}),
            dcc.Graph(figure=fig_heat,
                      style={'flex':'1'}),
        ], style={'display':'flex', 'gap':'16px',
                  'marginBottom':'16px'}),

        # Assumptions box
        html.Div([
            section_header("Model Assumptions"),
            html.P(
                "These are conservative industry estimates. "
                "Each is clearly stated and adjustable.",
                style={'color': COLORS['muted'],
                       'marginBottom': '12px',
                       'fontSize': '13px'}
            ),
            html.Div([
                html.Div([
                    html.Strong(f"{k}: ",
                                style={'color': COLORS['primary']}),
                    html.Span(
                        f"{v:,}" if isinstance(v, int)
                        else f"{v:.2%}" if isinstance(v, float)
                             and v < 1
                        else f"{v}",
                        style={'color': COLORS['text']}
                    ),
                ], style={'marginBottom': '6px',
                          'fontSize': '13px'})
                for k, v in assumptions.items()
                if k not in ['ndcg_random', 'ndcg_two_tower',
                             'relative_lift']
            ])
        ], style={
            'background'   : COLORS['surface'],
            'border'       : f"1px solid {COLORS['border']}",
            'borderRadius' : '12px',
            'padding'      : '20px',
        })
    ])


# ── Tab 6 — Bias & Fairness ────────────────────────────────

def build_tab_bias():
    bias   = DATA.get('bias', {})
    movies = DATA['movies']
    train  = DATA['train']

    pop_bias   = bias.get('popularity_bias', {})
    cold_start = bias.get('cold_start',      {})

    top_1  = pop_bias.get('top_1pct_rating_share',  0)
    top_5  = pop_bias.get('top_5pct_rating_share',  0)
    top_10 = pop_bias.get('top_10pct_rating_share', 0)
    cs_pct = cold_start.get('cold_start_pct',       0)
    cs_n   = cold_start.get('cold_start_movies_in_test', 0)

    # Popularity concentration
    fig_pop = go.Figure(go.Bar(
        x = ['Top 1%\nMovies', 'Top 5%\nMovies',
             'Top 10%\nMovies'],
        y = [top_1*100, top_5*100, top_10*100],
        marker_color = [COLORS['danger'],
                        COLORS['warning'],
                        COLORS['primary']],
        text  = [f'{v*100:.1f}%' for v in
                 [top_1, top_5, top_10]],
        textposition = 'outside',
    ))
    fig_pop.update_layout(
        title       = 'Popularity Concentration — Rating Share',
        yaxis_title = '% of Total Ratings',
        xaxis_title = 'Movie Population',
        **PLOTLY_TEMPLATE['layout']
    )

    # Genre representation
    movies['genre_list'] = movies['genres_clean'].str.split()

    all_genres    = {}
    for genres in movies['genre_list'].dropna():
        for g in genres:
            all_genres[g] = all_genres.get(g, 0) + 1

    train_movie_ids = set(train['movie_idx'].unique())
    rated_movies    = movies[
        movies['movie_idx'].isin(train_movie_ids)
    ]
    rated_genres    = {}
    for genres in rated_movies['genre_list'].dropna():
        for g in genres:
            rated_genres[g] = rated_genres.get(g, 0) + 1

    total_cat   = sum(all_genres.values())
    total_rated = sum(rated_genres.values())

    genre_bias_df = pd.DataFrame([
        {
            'genre'   : g,
            'cat_pct' : all_genres.get(g, 0) / max(total_cat, 1),
            'rate_pct': rated_genres.get(g, 0) / max(total_rated, 1),
        }
        for g in all_genres
    ])
    genre_bias_df['ratio'] = (
        genre_bias_df['rate_pct'] /
        genre_bias_df['cat_pct'].replace(0, np.nan)
    ).fillna(0)
    genre_bias_df = genre_bias_df.sort_values(
        'ratio', ascending=True
    ).head(18)

    colors_genre = [
        COLORS['success'] if r > 1.1
        else COLORS['danger'] if r < 0.9
        else COLORS['primary']
        for r in genre_bias_df['ratio']
    ]

    fig_genre = go.Figure(go.Bar(
        x            = genre_bias_df['ratio'],
        y            = genre_bias_df['genre'],
        orientation  = 'h',
        marker_color = colors_genre,
        text         = [f'{v:.2f}x' for v in
                        genre_bias_df['ratio']],
        textposition = 'outside',
    ))
    fig_genre.add_vline(
        x=1.0, line_dash='dash',
        line_color=COLORS['muted'],
        annotation_text='Fair representation',
        annotation_position='top'
    )
    fig_genre.update_layout(
        title       = 'Genre Representation Ratio\n'
                      'Green=over, Red=under, Blue=fair',
        xaxis_title = 'Ratio (rated share / catalogue share)',
        **PLOTLY_TEMPLATE['layout']
    )

    # Cold start summary
    fig_cold = go.Figure(go.Pie(
        values    = [100 - cs_pct*100, cs_pct*100],
        labels    = ['Warm movies\n(in training)',
                     'Cold start\n(unseen)'],
        hole      = 0.5,
        marker    = dict(colors=[
            COLORS['success'], COLORS['warning']
        ]),
        textinfo  = 'label+percent',
        textfont  = dict(color=COLORS['text']),
    ))
    fig_cold.update_layout(
        title = f'Cold Start Movies in Test Set: '
                f'{cs_n:,} ({cs_pct:.1%})',
        annotations=[dict(
            text=f'{cs_pct:.0%}\ncold',
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False,
            font=dict(color=COLORS['warning'])
        )],
        **PLOTLY_TEMPLATE['layout']
    )

    return html.Div([
        section_header(
            "Bias & Fairness",
            "Popularity bias, genre representation, "
            "and cold start analysis"
        ),

        # KPI row
        html.Div([
            kpi_card(
                f"{top_1:.1%}",
                "Top 1% Movies → Rating Share",
                COLORS['danger']
            ),
            kpi_card(
                f"{top_10:.1%}",
                "Top 10% Movies → Rating Share",
                COLORS['warning']
            ),
            kpi_card(
                f"{cs_pct:.1%}",
                "Cold Start Movies in Test",
                COLORS['warning']
            ),
            kpi_card(
                f"{cs_n:,}",
                "Cold Start Movie Count",
                COLORS['muted']
            ),
        ], style={'display':'flex', 'gap':'12px',
                  'marginBottom':'24px'}),

        html.Div([
            dcc.Graph(figure=fig_pop,   style={'flex':'1'}),
            dcc.Graph(figure=fig_cold,  style={'flex':'1'}),
        ], style={'display':'flex', 'gap':'16px',
                  'marginBottom':'16px'}),

        dcc.Graph(figure=fig_genre),

        # Mitigation note
        html.Div([
            section_header("Bias Mitigation Strategy"),
            html.Ul([
                html.Li(
                    "Popularity bias: Content tower (DistilBERT) "
                    "provides signal for cold start movies "
                    "independent of rating counts.",
                    style={'color':COLORS['text'],
                           'marginBottom':'8px'}
                ),
                html.Li(
                    "Genre bias: Over-represented genres (Drama, "
                    "Comedy) dominate training — niche genres get "
                    "fewer training signals.",
                    style={'color':COLORS['text'],
                           'marginBottom':'8px'}
                ),
                html.Li(
                    "Cold start: 12.93% of test movies were never "
                    "seen in training. Content tower partially "
                    "handles these via title+genre+genome tags.",
                    style={'color':COLORS['text'],
                           'marginBottom':'8px'}
                ),
                html.Li(
                    "Future work: Re-sampling strategies, "
                    "genre-aware loss functions, or "
                    "debiasing post-processing.",
                    style={'color':COLORS['muted'],
                           'marginBottom':'8px'}
                ),
            ], style={'paddingLeft':'20px'})
        ], style={
            'background'   : COLORS['surface'],
            'border'       : f"1px solid {COLORS['border']}",
            'borderRadius' : '12px',
            'padding'      : '20px',
            'marginTop'    : '16px',
        })
    ])


# ══════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    title = "Cinemate Dashboard",
    suppress_callback_exceptions = True
)

app.layout = html.Div([

    # ── Header ─────────────────────────────────────────────
    html.Div([
        html.Div([
            html.H1("🎬 Cinemate",
                    style={'margin':'0',
                           'fontSize':'24px',
                           'fontWeight':'700',
                           'color': COLORS['text']}),
            html.P(
                "Two-Tower Hybrid Recommender · Analytics Dashboard",
                style={'margin':'4px 0 0 0',
                       'fontSize':'13px',
                       'color': COLORS['muted']}
            ),
        ]),
        html.Div([
            html.Span("MovieLens 33M",
                      style={'background': '#1E1A3A',
                             'color': COLORS['primary'],
                             'padding': '4px 12px',
                             'borderRadius': '6px',
                             'fontSize': '12px',
                             'marginRight': '8px'}),
            html.Span("NDCG@10: 0.1235",
                      style={'background': '#1A2A1E',
                             'color': COLORS['success'],
                             'padding': '4px 12px',
                             'borderRadius': '6px',
                             'fontSize': '12px',
                             'marginRight': '8px'}),
            html.Span(
                f"Updated: "
                f"{datetime.now().strftime('%b %d, %Y')}",
                style={'color': COLORS['muted'],
                       'fontSize': '12px'}
            ),
        ])
    ], style={
        'background'   : COLORS['surface'],
        'borderBottom' : f"1px solid {COLORS['border']}",
        'padding'      : '16px 24px',
        'display'      : 'flex',
        'justifyContent': 'space-between',
        'alignItems'   : 'center',
    }),

    # ── Tabs ───────────────────────────────────────────────
    dcc.Tabs(
        id    = 'tabs',
        value = 'tab-dataset',
        children=[
            dcc.Tab(label='📊 Dataset',
                    value='tab-dataset'),
            dcc.Tab(label='🤖 Models',
                    value='tab-models'),
            dcc.Tab(label='🎯 Recommendations',
                    value='tab-recs'),
            dcc.Tab(label='🧪 A/B Test',
                    value='tab-ab'),
            dcc.Tab(label='💰 Business Impact',
                    value='tab-business'),
            dcc.Tab(label='⚖️ Bias & Fairness',
                    value='tab-bias'),
        ],
        style={'borderBottom': f'1px solid {COLORS["border"]}'},
        colors={
            'border'    : COLORS['border'],
            'primary'   : COLORS['primary'],
            'background': COLORS['surface'],
        }
    ),

    # ── Tab Content ────────────────────────────────────────
    html.Div(
        id='tab-content',
        style={'padding': '24px', 'background': COLORS['bg'],
               'minHeight': 'calc(100vh - 120px)'}
    ),

], style={
    'fontFamily'    : 'system-ui, -apple-system, sans-serif',
    'background'    : COLORS['bg'],
    'minHeight'     : '100vh',
})


# ══════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════

@callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_tab(tab):
    if tab == 'tab-dataset':
        return build_tab_dataset()
    elif tab == 'tab-models':
        return build_tab_models()
    elif tab == 'tab-recs':
        return build_tab_rec_analysis()
    elif tab == 'tab-ab':
        return build_tab_ab()
    elif tab == 'tab-business':
        return build_tab_business()
    elif tab == 'tab-bias':
        return build_tab_bias()
    return html.Div("Tab not found")


# ══════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Starting Cinemate Dashboard...")
    print("Visit: http://localhost:8050")
    app.run(debug=True, port=8050)