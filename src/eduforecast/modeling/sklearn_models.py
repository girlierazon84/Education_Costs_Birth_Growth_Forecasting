"""src/eduforecast/modeling/sklearn_models.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SklearnAutoregressorWrapper:
    """
    Adapter som transformerar en scikit-learn-regressor (t.ex. LinearRegression, RandomForestRegressor, Ridge) till en ren tidsseriemodell.

    Den implementerar gränssnittet .predict(steps=...) via autoregressiv
    steg-för-steg-prediktion (Recursive Multi-step Forecasting).
    """
    model: Any
    lags: int
    last_historical_values: np.ndarray

    def predict(self, steps: int) -> np.ndarray:
        """
        Genererar en prognos för 'steps' antal framtida tidssteg genom att
        rekursivt mata in sina egna prediktioner som framtida features.
        """
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps måste vara >= 0")
        if steps == 0:
            return np.empty(0, dtype=float)

        if self.last_historical_values.size < self.lags:
            raise ValueError(
                f"För få historiska värden för att bygga start-lag."
                f"Kräver {self.lags}, fick {self.last_historical_values.size}."
            )

        # Skapa en rullande buffer initierad med de sista kända historiska värdena.
        # Vi tar exakt de sista 'lags' värdena och behåller tidsordningen (äldst till nyast).
        current_window = self.last_historical_values[-self.lags:].tolist()
        predictions = []

        for _ in range(steps):
            # Skapa X-vektor för nuvarande steg. Formen måste matcha scikit.learn (1, n_features)
            X_curr = np.array([current_window], dtype=float)

            # Generera prediktion för nästa år (t.ex. 2024)
            y_pred = float(self.model.predict(X_curr)[0])
            predictions.append(y_pred)

            # Uppdatera det rullande fönstret: ta bort det äldsta värdet och lägg till det nya
            current_window.pop(0)
            current_window.append(y_pred)

        return np.array(predictions, dtype=float)

    def fit_sklearn_autoregressor(
        model: Any,
        y_train: np.ndarray,
        lags: int = 3
    ) -> SklearnAutoregressorWrapper:
        """
        Hjälpfunktion för att transformera en tidsserie till en matris med lag-features,
        träna scikit-learn-modellen och returnera den färdiga adaptern.
        """
        y = np.asarray(y_train, dtype=float)
        if y.size <= lags:
            raise ValueError(
                f"Tidsserien är för kort ({y.size}) för antal önskade lags ({lags})."
            )

        # Bygg X (features) och y (targets) från tidsserien
        # Exempel om lags=3: X[y1, y2, y3] -> target=y4
        X_list = []
        y_list = []

        for i in range(len(y) - lags):
            X_list.append(y[i : i + lags])
            y_list.append(y[i + lags])

        X_train = np.array(X_list, dtype=float)
        y_train_target = np.array(y_list, dtype=float)

        # Träna scikit-lern-modellen på plats
        model.fit(X_train, y_train_target)

        # Returnera adaptern laddad med modellen och de sista kända värdena från 2023
        return SklearnAutoregressorWrapper(
            model=model,
            lags=lags,
            last_historical_values=y
        )

