\name{predict.stsmSS}
\alias{predict.stsmSS}

\title{Kalman Filter for State Space Models}

\description{
These functions run the iterative equations of the Kalman filter 
for a state space model.
}

\usage{
\S3method{predict}{stsmSS}(object, y, n.ahead = 12L, ...)
}

\arguments{
\item{object}{a list containing the matrices of the state space model.}
\item{y}{a numeric time series.}
\item{n.ahead}{a numeric. The number of steps ahead to predict.}
\item{...}{further arguments. Currently omitted.}
}

\value{
A list containing the following elements:
item{pred}{a time series containing \code{n.ahead} predictions.}
item{se}{a time series containing the standard errors of \code{pred}.}
item{a}{a univariate or multivariate time series object containing \code{n.ahead} 
predictions for the state vector.}
item{P}{a univariate or multivariate time series object containing the square of 
the standard errors of \code{a}.}
}

\details{
This function computes the same values as the 
function \link[stats]{predict.StructTS} from the \pkg{stats} package but 
the predictions of the components are also returned.
}

\references{
Harvey, A. C. (1989).
\emph{Forecasting, Structural Time Series Models and the Kalman Filter}. 
Cambridge University Press.
}

\examples{
## local level model
## Nile time series
y <- Nile
m <- stsm.class::stsm.model(model = "local-level", y = y, transPars = "StructTS")
fit <- StructTS(y, "level")
m <- stsm.class::set.pars(m, as.vector(fit$coef[c(2,1)]) * 100 / var(y))
ss <- stsm.class::char2numeric(m, P0cov = TRUE)
res <- predict(ss, y, 5)

# display forecasts and confidence intervals
plot(cbind(y, res$pred), type = "n", plot.type = "single")
lines(y)
lines(res$pred, col = "blue")
lines(res$pred + 2 * res$se, col = "red", lty = 2)
lines(res$pred - 2 * res$se, col = "red", lty = 2)

# for the whole series, the above is the same as "predict.StructTS" 
all.equal(res$pred, predict(fit, 5)$pred)
all.equal(res$se, predict(fit, 5)$se)

## basic Structural model
## AirPassengers time series (in logarithms)
y <- log(AirPassengers)
m <- stsm.class::stsm.model(model = "BSM", y = y, transPars = "StructTS")
fit <- StructTS(y, "BSM")
m <- stsm.class::set.pars(m, as.vector(fit$coef[c(4,1:3)]) * 100 / var(y))
ss <- stsm.class::char2numeric(m, P0cov = TRUE)
res <- predict(ss, y, 12)

all.equal(res$pred, predict(fit, 12)$pred)
all.equal(res$se, predict(fit, 12)$se)

# forecasts and confidence intervals for the series
# scaled back to original scale
expy <- exp(y)
plot(cbind(expy, exp(res$pred + 2 * res$se)), type = "n", plot.type = "single")
lines(expy)
lines(exp(res$pred), col = "blue")
lines(exp(res$pred + 2 * res$se), col = "red", lty = 2)
lines(exp(res$pred - 2 * res$se), col = "red", lty = 2)

# forecasts for the trend component
# the aproach in StructTS() seems to seasonal fluctuations in the trend
# see the "stsm" package for a more flexible interface for maximum likelihood 
# procedures to fit a structural time series model
trend <- exp(fitted(fit)[,1])
plot(cbind(trend, AirPassengers), 
  type = "n", plot.type = "single")
lines(AirPassengers, col = "gray")
lines(trend)
lines(exp(res$a[,1]), col = "blue")
lines(exp(res$a[,1] + 2 * sqrt(res$P[,1])), col = "red", lty = 2)
lines(exp(res$a[,1] - 2 * sqrt(res$P[,1])), col = "red", lty = 2)

# forecasts for the seasonal component
seas <- exp(fitted(fit)[,3])
plot(cbind(seas, exp(res$a[,3]) + 2 * sqrt(res$P[,3])), 
  type = "n", plot.type = "single")
lines(seas)
lines(exp(res$a[,3]), col = "blue")
lines(exp(res$a[,3] + 2 * sqrt(res$P[,3])), col = "red", lty = 2)
lines(exp(res$a[,3] - 2 * sqrt(res$P[,3])), col = "red", lty = 2)
}

\keyword{ts, model}
