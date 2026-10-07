import re
import uuid
import joblib
import pandas as pd
import gradio as gr
import html
import os

from scipy.sparse import hstack


# ==============================
# Load trained models
# ==============================

tfidf = joblib.load("models/word_tfidf.joblib")
char_tfidf = joblib.load("models/char_tfidf.joblib")
combined_model = joblib.load("models/combined_model.joblib")

print("Models loaded successfully!")

# ==============================
# Text preprocessing
# ==============================

def replace_placeholders(text):
    return re.sub(
        r"\{\{(.*?)\}\}",
        lambda match: match.group(1).replace(" ", "_").upper(),
        text
    )

# ==============================
# Intent prediction
# ==============================

CONFIDENCE_THRESHOLD = 0.70


def predict_intent_with_confidence_combined(message):
    cleaned_message = message.lower()
    cleaned_message = replace_placeholders(cleaned_message)

    word_vector = tfidf.transform([cleaned_message])
    char_vector = char_tfidf.transform([cleaned_message])

    combined_vector = hstack([
        word_vector,
        char_vector
    ])

    probabilities = combined_model.predict_proba(combined_vector)[0]

    predicted_index = probabilities.argmax()
    predicted_intent = combined_model.classes_[predicted_index]
    probability = probabilities[predicted_index]

    if probability >= CONFIDENCE_THRESHOLD:
        decision = "AUTO_ROUTE"
    else:
        decision = "HUMAN_REVIEW"

    return predicted_intent, probability, decision

# ==============================
# Intent → Support Team mapping
# ==============================

intent_to_team = {
    "cancel_order": "Order Support",
    "change_order": "Order Support",
    "place_order": "Order Support",
    "track_order": "Order Support",

    "change_shipping_address": "Delivery Support",
    "set_up_shipping_address": "Delivery Support",
    "delivery_options": "Delivery Support",
    "delivery_period": "Delivery Support",

    "payment_issue": "Payment Support",
    "check_payment_methods": "Payment Support",

    "get_refund": "Refund Support",
    "track_refund": "Refund Support",
    "check_refund_policy": "Refund Support",

    "check_invoice": "Billing Support",
    "get_invoice": "Billing Support",

    "create_account": "Account Support",
    "delete_account": "Account Support",
    "edit_account": "Account Support",
    "recover_password": "Account Support",
    "registration_problems": "Account Support",
    "switch_account": "Account Support",

    "contact_customer_service": "Customer Service",
    "contact_human_agent": "Customer Service",

    "complaint": "Customer Service",
    "review": "Customer Service",

    "check_cancellation_fee": "Order Support",
    "newsletter_subscription": "Subscription Support"
}

# ==============================
# Priority prediction
# ==============================

def predict_priority(message):
    message_lower = message.lower()

    high_priority_keywords = [
        "urgent",
        "urgently",
        "asap",
        "immediately",
        "charged twice",
        "charged multiple times",
        "money deducted",
        "payment deducted",
        "account locked",
        "can't access",
        "cannot access",
        "fraud",
        "unauthorized",
        "stolen",
        "security issue"
    ]

    medium_priority_keywords = [
        "refund",
        "cancel",
        "complaint",
        "payment issue",
        "not received",
        "delayed",
        "problem",
        "error",
        "failed"
    ]

    for keyword in high_priority_keywords:
        if keyword in message_lower:
            return "HIGH"

    for keyword in medium_priority_keywords:
        if keyword in message_lower:
            return "MEDIUM"

    return "LOW"

# ==============================
# Complete ticket processing
# ==============================

def process_ticket(message):

    predicted_intent, confidence, decision = (
        predict_intent_with_confidence_combined(message)
    )

    predicted_team = intent_to_team[predicted_intent]

    priority = predict_priority(message)

    if decision == "AUTO_ROUTE":
        routing_destination = predicted_team
    else:
        routing_destination = "Human Review Queue"

    ticket_id = "TKT-" + str(uuid.uuid4())[:8].upper()

    return {
        "ticket_id": ticket_id,
        "message": message,
        "predicted_intent": predicted_intent,
        "confidence": round(confidence, 4),
        "priority": priority,
        "decision": decision,
        "predicted_team": predicted_team,
        "routing_destination": routing_destination
    }

# -------------------------------------------------------------
# --------------------------------------------------------------------------
# -----------------------------------------------------------------------

# ==============================
# V4 Dashboard + Ticket Queue
# ==============================

def make_result_card(result):
    confidence_percent = result["confidence"] * 100

    if result["decision"] == "AUTO_ROUTE":
        decision_color = "#22c55e"
        decision_icon = "✅"
    else:
        decision_color = "#f59e0b"
        decision_icon = "👤"

    if result["priority"] == "HIGH":
        priority_color = "#ef4444"
    elif result["priority"] == "MEDIUM":
        priority_color = "#f59e0b"
    else:
        priority_color = "#22c55e"

    return f"""
    <div style="
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 14px;
        padding: 24px;
        color: #f9fafb;
        margin-top: 10px;
    ">

        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            margin-bottom:20px;
        ">
            <h2 style="margin:0; color:#f9fafb;">
                🎫 Ticket Analysis
            </h2>

            <span style="
                background:{decision_color};
                color:white;
                padding:7px 14px;
                border-radius:20px;
                font-weight:600;
            ">
                {decision_icon} {result["decision"].replace("_", " ")}
            </span>
        </div>

        <div style="
            background:#111827;
            border-radius:10px;
            padding:15px;
            margin-bottom:15px;
        ">
            <div style="color:#9ca3af; font-size:13px;">
                TICKET ID
            </div>
            <div style="font-size:18px; font-weight:600;">
                {result["ticket_id"]}
            </div>
        </div>

        <div style="
            background:#111827;
            border-radius:10px;
            padding:15px;
            margin-bottom:15px;
        ">
            <div style="color:#9ca3af; font-size:13px;">
                CUSTOMER MESSAGE
            </div>
            <div style="font-size:16px; margin-top:6px;">
                {html.escape(result["message"])}
            </div>
        </div>

        <div style="
            display:grid;
            grid-template-columns:repeat(2, 1fr);
            gap:12px;
        ">

            <div style="
                background:#111827;
                padding:15px;
                border-radius:10px;
            ">
                <div style="color:#9ca3af; font-size:13px;">
                    PREDICTED INTENT
                </div>
                <div style="font-size:17px; font-weight:600; margin-top:5px;">
                    {result["predicted_intent"]}
                </div>
            </div>

            <div style="
                background:#111827;
                padding:15px;
                border-radius:10px;
            ">
                <div style="color:#9ca3af; font-size:13px;">
                    SUPPORT TEAM
                </div>
                <div style="font-size:17px; font-weight:600; margin-top:5px;">
                    {result["predicted_team"]}
                </div>
            </div>

            <div style="
                background:#111827;
                padding:15px;
                border-radius:10px;
            ">
                <div style="color:#9ca3af; font-size:13px;">
                    CONFIDENCE
                </div>

                <div style="font-size:20px; font-weight:700; margin-top:5px;">
                    {confidence_percent:.2f}%
                </div>

                <div style="
                    background:#374151;
                    height:8px;
                    border-radius:10px;
                    margin-top:8px;
                    overflow:hidden;
                ">
                    <div style="
                        width:{confidence_percent}%;
                        height:100%;
                        background:#f97316;
                        border-radius:10px;
                    "></div>
                </div>
            </div>

            <div style="
                background:#111827;
                padding:15px;
                border-radius:10px;
            ">
                <div style="color:#9ca3af; font-size:13px;">
                    PRIORITY
                </div>

                <div style="
                    color:{priority_color};
                    font-size:18px;
                    font-weight:700;
                    margin-top:5px;
                ">
                    {result["priority"]}
                </div>
            </div>

        </div>

        <div style="
            background:#111827;
            padding:15px;
            border-radius:10px;
            margin-top:12px;
        ">
            <div style="color:#9ca3af; font-size:13px;">
                ROUTING DESTINATION
            </div>

            <div style="
                font-size:18px;
                font-weight:600;
                margin-top:5px;
            ">
                📍 {result["routing_destination"]}
            </div>
        </div>

    </div>
    """


def update_stats(history):

    total = len(history)

    auto_route = sum(
        1 for ticket in history
        if ticket["decision"] == "AUTO_ROUTE"
    )

    human_review = sum(
        1 for ticket in history
        if ticket["decision"] == "HUMAN_REVIEW"
    )

    high = sum(
        1 for ticket in history
        if ticket["priority"] == "HIGH"
    )

    medium = sum(
        1 for ticket in history
        if ticket["priority"] == "MEDIUM"
    )

    low = sum(
        1 for ticket in history
        if ticket["priority"] == "LOW"
    )

    def card(title, value, icon):
        return f"""
        <div style="
            background:#1f2937;
            border:1px solid #374151;
            border-radius:12px;
            padding:18px;
            text-align:center;
        ">
            <div style="font-size:25px;">{icon}</div>
            <div style="
                color:#9ca3af;
                font-size:13px;
                margin-top:5px;
            ">
                {title}
            </div>
            <div style="
                font-size:25px;
                font-weight:700;
                margin-top:5px;
                color:#f9fafb;
            ">
                {value}
            </div>
        </div>
        """

    return (
        card("TOTAL TICKETS", total, "🎫"),
        card("AUTO ROUTED", auto_route, "🤖"),
        card("HUMAN REVIEW", human_review, "👤"),
        card("HIGH PRIORITY", high, "🔴"),
        card("MEDIUM PRIORITY", medium, "🟠"),
        card("LOW PRIORITY", low, "🟢")
    )


def history_to_dataframe(history):

    if not history:
        return pd.DataFrame(
            columns=[
                "Ticket ID",
                "Message",
                "Intent",
                "Confidence",
                "Priority",
                "Decision",
                "Support Team",
                "Routing"
            ]
        )

    rows = []

    for ticket in history:
        rows.append({
            "Ticket ID": ticket["ticket_id"],
            "Message": ticket["message"],
            "Intent": ticket["predicted_intent"],
            "Confidence": f'{ticket["confidence"]:.2%}',
            "Priority": ticket["priority"],
            "Decision": ticket["decision"],
            "Support Team": ticket["predicted_team"],
            "Routing": ticket["routing_destination"]
        })

    return pd.DataFrame(rows)


def analyze_ticket_v4(message, history):

    if history is None:
        history = []

    if not message or not message.strip():

        stats = update_stats(history)

        return (
            """
            <div style="
                background:#3f1d1d;
                border:1px solid #ef4444;
                border-radius:12px;
                padding:18px;
                color:#fca5a5;
            ">
                ⚠️ Please enter a customer support ticket.
            </div>
            """,
            history,
            history_to_dataframe(history),
            *stats
        )

    result = process_ticket(message.strip())

    history = history + [result]

    stats = update_stats(history)

    return (
        make_result_card(result),
        history,
        history_to_dataframe(history),
        *stats
    )


def filter_tickets(history, search, priority, decision, team):

    if not history:
        return history_to_dataframe([])

    filtered = history.copy()

    # Search
    if search and search.strip():

        search_text = search.strip().lower()

        filtered = [
            ticket for ticket in filtered
            if (
                search_text in ticket["ticket_id"].lower()
                or search_text in ticket["message"].lower()
                or search_text in ticket["predicted_intent"].lower()
            )
        ]

    # Priority
    if priority != "ALL":
        filtered = [
            ticket for ticket in filtered
            if ticket["priority"] == priority
        ]

    # Decision
    if decision != "ALL":
        filtered = [
            ticket for ticket in filtered
            if ticket["decision"] == decision
        ]

    # Support team
    if team != "ALL":
        filtered = [
            ticket for ticket in filtered
            if ticket["predicted_team"] == team
        ]

    return history_to_dataframe(filtered)


def reset_filters(history):

    return (
        "",
        "ALL",
        "ALL",
        "ALL",
        history_to_dataframe(history)
    )


def clear_all():

    empty_history = []

    stats = update_stats(empty_history)

    return (
        empty_history,
        history_to_dataframe(empty_history),
        *stats
    )


# ==============================
# Gradio Interface
# ==============================

teams = sorted(set(intent_to_team.values()))

with gr.Blocks(
    title="AI Customer Support Ticket Classifier V4"
) as demo:

    gr.Markdown(
        """
        # 🤖 AI Customer Support Ticket Classifier V4

        ### Intelligent ticket classification, priority detection and routing
        """
    )

    # --------------------------
    # Ticket Analyzer
    # --------------------------

    with gr.Row():

        with gr.Column(scale=1):

            ticket_input = gr.Textbox(
                label="Customer Support Ticket",
                placeholder="Enter a customer support message...",
                lines=5
            )

            analyze_button = gr.Button(
                "🔍 Analyze Ticket",
                variant="primary"
            )

            clear_button = gr.Button(
                "🧹 Clear"
            )

        with gr.Column(scale=1):

            result_output = gr.HTML(
                value="""
                <div style="
                    background:#1f2937;
                    border:1px solid #374151;
                    border-radius:12px;
                    padding:25px;
                    color:#9ca3af;
                    text-align:center;
                ">
                    🎫 Enter a customer ticket and click
                    <b>Analyze Ticket</b>
                </div>
                """
            )

    # --------------------------
    # Dashboard
    # --------------------------

    gr.Markdown("## 📊 Ticket Dashboard")

    with gr.Row():

        total_stats = gr.HTML()
        auto_stats = gr.HTML()
        review_stats = gr.HTML()

    with gr.Row():

        high_stats = gr.HTML()
        medium_stats = gr.HTML()
        low_stats = gr.HTML()

    # --------------------------
    # Ticket Queue
    # --------------------------

    gr.Markdown("## 🔎 Ticket Queue")

    with gr.Row():

        search_input = gr.Textbox(
            label="Search",
            placeholder="Search message, intent or ticket ID..."
        )

        priority_filter = gr.Dropdown(
            choices=["ALL", "HIGH", "MEDIUM", "LOW"],
            value="ALL",
            label="Priority"
        )

        decision_filter = gr.Dropdown(
            choices=["ALL", "AUTO_ROUTE", "HUMAN_REVIEW"],
            value="ALL",
            label="Decision"
        )

        team_filter = gr.Dropdown(
            choices=["ALL"] + teams,
            value="ALL",
            label="Support Team"
        )

    with gr.Row():

        apply_filter_button = gr.Button(
            "🔍 Apply Filters",
            variant="primary"
        )

        reset_filter_button = gr.Button(
            "🔄 Reset Filters"
        )

    ticket_table = gr.Dataframe(
        headers=[
            "Ticket ID",
            "Message",
            "Intent",
            "Confidence",
            "Priority",
            "Decision",
            "Support Team",
            "Routing"
        ],
        datatype=[
            "str",
            "str",
            "str",
            "str",
            "str",
            "str",
            "str",
            "str"
        ],
        value=history_to_dataframe([]),
        interactive=False,
        wrap=True
    )

    clear_all_button = gr.Button(
        "🗑️ Clear All Tickets"
    )

    # --------------------------
    # Session State
    # --------------------------

    ticket_history = gr.State([])

    # --------------------------
    # Event Handlers
    # --------------------------

    analyze_button.click(
        fn=analyze_ticket_v4,
        inputs=[
            ticket_input,
            ticket_history
        ],
        outputs=[
            result_output,
            ticket_history,
            ticket_table,
            total_stats,
            auto_stats,
            review_stats,
            high_stats,
            medium_stats,
            low_stats
        ]
    )

    clear_button.click(
        fn=lambda: "",
        inputs=None,
        outputs=ticket_input
    )

    apply_filter_button.click(
        fn=filter_tickets,
        inputs=[
            ticket_history,
            search_input,
            priority_filter,
            decision_filter,
            team_filter
        ],
        outputs=ticket_table
    )

    reset_filter_button.click(
        fn=reset_filters,
        inputs=ticket_history,
        outputs=[
            search_input,
            priority_filter,
            decision_filter,
            team_filter,
            ticket_table
        ]
    )

    clear_all_button.click(
        fn=clear_all,
        inputs=None,
        outputs=[
            ticket_history,
            ticket_table,
            total_stats,
            auto_stats,
            review_stats,
            high_stats,
            medium_stats,
            low_stats
        ]
    )


# ==============================
# Launch
# ==============================

if __name__ == "__main__":
    demo.launch()



# ==============================
# To run it in terminal
# ==============================

# cd "C:\Users\Si11191Sr\AI-Customer-Support-Ticket-Classifier"
# python app.py