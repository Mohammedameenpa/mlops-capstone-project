from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChurnFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")


    age: int = Field(ge=18, le=78)
    service_segment: Literal[
        "Cloud Storage",
        "Fintech",
        "Fitness",
        "Gaming",
        "News & Media",
        "Productivity",
        "Streaming",
    ]
    signup_channel: Literal[
        "App Store",
        "Email Campaign",
        "Influencer",
        "Organic Search",
        "Paid Social",
        "Referral",
    ]
    primary_device: Literal["Desktop", "Mobile", "Smart TV", "Tablet"]
    tenure_months: int = Field(ge=1, le=84)
    contract_type: Literal["Annual", "Month-to-month", "Two year"]
    plan_tier: Literal["Basic", "Plus", "Premium", "Standard"]
    num_services: int = Field(ge=1, le=8)
    has_family_plan: bool
    monthly_charges: float = Field(ge=5.25, le=61.93)
    total_charges: float = Field(ge=7.45, le=3410.54)
    payment_method: Literal[
        "Bank transfer",
        "Buy now pay later",
        "Credit card",
        "Debit card",
        "Digital wallet",
        "UPI / Instant transfer",
    ]
    auto_pay: bool
    payment_failures_last_year: int = Field(ge=0, le=5)
    support_tickets_last_year: int = Field(ge=0, le=10)
    avg_resolution_hours: float = Field(ge=0.5, le=59.5)
    uses_mobile_app: bool
    app_logins_per_month: int = Field(ge=0, le=36)
    avg_session_minutes: float = Field(ge=0.5, le=73.4)
    monthly_usage_hours: float = Field(ge=0.0, le=26.9)
    days_since_last_login: int = Field(ge=0, le=150)
    email_open_rate_pct: float = Field(ge=0.4, le=99.7)
    satisfaction_score: float = Field(ge=4.0, le=10.0)
    nps_score: int = Field(ge=1, le=10)
    competitor_offer_received: bool
    price_increase_last_year_pct: float = Field(ge=0.0, le=30.0)
    discount_offered: bool
    discount_pct: float = Field(ge=0.0, le=35.8)
    referrals_made: int = Field(ge=0, le=8)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: ChurnFeatures


class PredictResponse(BaseModel):
    model_name: str
    model_version: int
    churn: bool
    churn_probability: float = Field(ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: int
