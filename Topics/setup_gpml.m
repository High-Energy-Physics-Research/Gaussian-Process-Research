function setup_gpml()
% Caminho absoluto do arquivo setup_gpml.m
root = fileparts(mfilename('fullpath'));

% Caminho do submodule GPML
gpml = fullfile(root, '..', 'external', 'gpml');

% Adiciona GPML e subpastas essenciais
addpath(gpml)
addpath(fullfile(gpml,'cov'))
addpath(fullfile(gpml,'mean'))
addpath(fullfile(gpml,'lik'))
addpath(fullfile(gpml,'inf'))
addpath(fullfile(gpml,'util'))
addpath(fullfile(gpml,'att'))
addpath(fullfile(gpml,'prior'))
addpath(fullfile(gpml,'octavepkg'))


% (opcional) confirmação visual
fprintf('GPML loaded from:\n%s\n', gpml);
end