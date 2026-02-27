//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=1> n;
  array[n] int<lower=0, upper=1> h;
}

// The parameters accepted by the model.
// This is the random one (random rate)
parameters {
  real<lower=0, upper=1> theta;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  target += beta_lpdf(theta | 1,1);
  target += bernoulli_lpmf(h | theta);
}

#the last bit, for getting the posteriors etc
generated quantities {
  real<lower=0, upper=1> theta_prior;
  real<lower=0, upper=1> theta_posterior;
  
  int<lower=0, upper=n> prior_preds;
  int<lower=0, upper=n> posterior_preds;
  
  theta_prior= inv_logit(normal_rng(0,1));
  theta_posterior= inv_logit(theta)
  
  prior_preds= binomial_rng(n, theta_prior); // n is the n of trials
  posterior_preds= binomial_rng(n, inv_logit(theta));
}


