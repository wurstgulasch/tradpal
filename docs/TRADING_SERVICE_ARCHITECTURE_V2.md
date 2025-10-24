# TradPal Trading Service Architecture

## Overview
The trading_service is organized as a microservices architecture with specialized services for different trading functions.

## Service Architecture

### Core Trading Services
```mermaid
graph TB
    subgraph "Trading Service Architecture"
        subgraph "Specialized AI/ML Services"
            ML[ml_training_service<br/>Port: 8011<br/>ML Model Training & Ensembles]
            RL[reinforcement_learning_service<br/>Port: 8012<br/>RL Agents & Decision Making]
            MR[market_regime_service<br/>Port: 8013<br/>Regime Detection & Analysis]
            RM[risk_management_service<br/>Port: 8014<br/>Risk Assessment & Position Sizing]
            TE[trading_execution_service<br/>Port: 8015<br/>Order Execution & Portfolio Mgmt]
        end

        subgraph "Orchestration Services"
            AI[trading_ai_service<br/>Port: 8003<br/>AI Orchestration & Strategy]
            BT[backtesting_service<br/>Port: 8004<br/>Historical Simulation]
            LIVE[trading_bot_live_service<br/>Port: 8005<br/>Live Trading Execution]
        end

        subgraph "Central Components"
            ORCH[orchestrator.py<br/>Service Coordination]
            CLIENT[central_service_client.py<br/>Cross-Service Communication]
        end
    end

    subgraph "External Dependencies"
        DATA[data_service<br/>Market Data & Sources]
        CORE[core_service<br/>Technical Indicators]
        SEC[security_service<br/>Authentication]
        EVENT[event_system_service<br/>Event Streaming]
    end

    %% Data Flow
    DATA --> ML
    DATA --> RL
    DATA --> MR
    CORE --> MR
    CORE --> RM

    ML --> AI
    RL --> AI
    MR --> AI
    MR --> RL
    RM --> AI

    AI --> TE
    TE --> LIVE
    TE --> BT

    ORCH --> AI
    ORCH --> BT
    ORCH --> LIVE

    CLIENT --> ORCH

    %% Security & Events
    SEC -.-> TE
    SEC -.-> DATA
    EVENT -.-> AI
    EVENT -.-> TE
```

## Service Responsibilities

### Specialized AI/ML Services
1. **ml_training_service** (Port 8011)
   - ML model training and validation
   - Ensemble method implementation
   - Model performance optimization
   - Feature engineering and selection

2. **reinforcement_learning_service** (Port 8012)
   - Q-Learning and Deep RL agents
   - Trading environment simulation
   - Reward function design
   - Experience replay and training

3. **market_regime_service** (Port 8013)
   - Market regime classification
   - Volatility and trend analysis
   - Multi-timeframe pattern recognition
   - Regime transition detection

4. **risk_management_service** (Port 8014)
   - Position sizing calculations
   - VaR and risk metrics
   - Kelly criterion optimization
   - Portfolio risk assessment

5. **trading_execution_service** (Port 8015)
   - Order execution and management
   - Portfolio position tracking
   - Broker API integration (CCXT)
   - Transaction cost optimization

### Orchestration Services
6. **trading_ai_service** (Port 8003)
   - AI-powered trading orchestration
   - Signal generation and aggregation
   - Strategy execution coordination
   - Performance monitoring

7. **backtesting_service** (Port 8004)
   - Historical simulation engine
   - Performance metrics calculation
   - Walk-forward validation
   - Strategy optimization

8. **trading_bot_live_service** (Port 8005)
   - Live trading execution
   - Real-time position management
   - Risk monitoring and stops
   - Emergency shutdown procedures

## Data Flow Architecture

### Signal Generation Pipeline
```mermaid
sequenceDiagram
    participant Data as data_service
    participant Core as core_service
    participant MR as market_regime_service
    participant ML as ml_training_service
    participant RL as reinforcement_learning_service
    participant RM as risk_management_service
    participant AI as trading_ai_service
    participant TE as trading_execution_service

    Data->>Core: Raw market data
    Core->>MR: Technical indicators
    MR->>ML: Regime-aware features
    MR->>RL: Market context
    ML->>AI: ML predictions
    RL->>AI: RL actions
    RM->>AI: Risk parameters
    AI->>TE: Trading signals
    TE->>TE: Order execution
```

### Backtesting Pipeline
```mermaid
sequenceDiagram
    participant Data as data_service
    participant BT as backtesting_service
    participant AI as trading_ai_service
    participant TE as trading_execution_service

    Data->>BT: Historical data
    BT->>AI: Simulation environment
    AI->>TE: Simulated signals
    TE->>BT: Simulated trades
    BT->>BT: Performance metrics
```

## Service Communication Patterns

### Synchronous Communication
- REST API calls between services
- Health checks and status queries
- Configuration synchronization

### Asynchronous Communication
- Redis Streams for event-driven updates
- Market data updates
- Trading signal propagation
- Performance metrics publishing

### Service Dependencies
- **ml_training_service**: Independent (uses data_service)
- **reinforcement_learning_service**: Independent (uses market_regime_service)
- **market_regime_service**: Independent (uses core_service)
- **risk_management_service**: Independent (uses core_service)
- **trading_execution_service**: Depends on data_service, security_service
- **trading_ai_service**: Orchestrates all specialized services
- **backtesting_service**: Depends on all trading services
- **trading_bot_live_service**: Depends on all trading services

## Configuration Management
- Service URLs defined in `config/service_settings.py`
- Environment-specific configurations via `.env` files
- Dynamic service discovery through API Gateway
- Centralized logging and monitoring

## Deployment Architecture
- Each service runs independently on dedicated ports (8001-8015)
- Docker containerization for each service
- Kubernetes orchestration for production
- Health checks and automatic restart policies

## Monitoring & Observability
- Prometheus metrics collection
- Grafana dashboards for visualization
- Centralized logging with correlation IDs
- Alert management through notification_service
- Performance profiling and bottleneck analysis

---
*Architecture Overview - TradPal Trading Service*
*Generated: October 24, 2025*