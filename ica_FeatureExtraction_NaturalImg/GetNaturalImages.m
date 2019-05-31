function [X, whiteningMatrix, dewhiteningMatrix] = ...
				GetNaturalImages( samples, winsize, rdim );
% data - gathers patches from the gray-scale images
%
% INPUT variables:
% samples            total number of patches to take
% winsize            patch width in pixels
% rdim               reduced dimensionality
%
% OUTPUT variables:
% X                  the whitened data as column vectors
% whiteningMatrix    transformation of patch-space to X-space
% dewhiteningMatrix  inverse transformation
%

rand('seed', 0);
  
%----------------------------------------------------------------------
% Gather rectangular image windows
%----------------------------------------------------------------------

% We have a total of 13 images, 画像の数:13枚
dataNum = 13;

% This is how many patches to take per image,　1枚からいくつパッチをとるか？
getsample = floor(samples/dataNum);

% Initialize the matrix to hold the patches, パッチ画像の初期化
X = zeros(winsize^2,getsample*dataNum);

% 切り出すパッチ数，初期値１
sampleNum = 1; 

% 画像枚数毎繰り返す
for i=(1:dataNum)

  % Even things out (take enough from last image)
  if i==dataNum
      % 端数にならないようにsamples数に正確になるように、最後の画像のみパッチ多く取る
      getsample = samples-sampleNum+1;
  end
  
  % Load the image, 画像のロード
  I = imread([ num2str(i) '.tiff']);

  % Normalize to zero mean and unit variance
  % 1枚の画像に対して，まず正規化をする
  I = double(I);
  I = I-mean(mean(I));          % 全画素から平均値を引く
  I = I/sqrt(mean(mean(I.^2))); % 分散が1になるように正規化する:規準化データ
  
  % Sample 
  fprintf('Sampling image %d...\n',i);
  sizex = size(I,2); sizey = size(I,1);
  posx = floor(rand(1,getsample)*(sizex-winsize-2))+1; % 乱数にて、画像からパッチの取得の場所を決める
  posy = floor(rand(1,getsample)*(sizey-winsize-1))+1; % 乱数にて、画像からパッチの取得の場所を決める
  
  for j=1:getsample
    % 縦:ピクセル数，256：１つのパッチデータ16*16。
    % 横にパッチ数が続く。1枚の画像からgetsample個取得し、画像枚数分続く。
    X(:,sampleNum) = reshape( I(posy(1,j):posy(1,j)+winsize-1, ...
			posx(1,j):posx(1,j)+winsize-1),[winsize^2 1]);
    sampleNum=sampleNum+1;
  end 
  
end

%----------------------------------------------------------------------
% Subtract local mean gray-scale value from each patch
%----------------------------------------------------------------------
fprintf('Subtracting local mean...\n');
% 各パッチごとに平均が０になるようにする
X = X-ones(size(X,1),1)*mean(X);

%----------------------------------------------------------------------
% Reduce the dimension and whiten at the same time!
%----------------------------------------------------------------------

% Calculate the eigenvalues and eigenvectors of covariance matrix.
fprintf ('Calculating covariance...\n');
% 分散共分散行列を求める。
covarianceMatrix = X*X'/size(X,2);
% 固有ベクトルE、固有値Dを求める。
[E, D] = eig(covarianceMatrix);

% Sort the eigenvalues and select subset, and whiten
fprintf('Reducing dimensionality and whitening...\n');
[dummy,order] = sort(diag(-D)); % Dが大きいのが上に来るようにソートする。
E = E(:,order(1:rdim));         % rdim(160)の数までとる。寄与率が大きい方から１６０個分
d = diag(D); 
d = real(d.^(-0.5));            % realにするのは-1にルートが入るため
D = diag(d(order(1:rdim)));     % Eにあわせえて、rdim(160)の数までとる。
X = D*E'*X;                     % 白色化行列をかける

whiteningMatrix = D*E';
dewhiteningMatrix = E*D^(-1);

%保存：
fname='whitening.mat';
fprintf(['Writing file: ' fname '...']);
eval(['save ' fname ' whiteningMatrix dewhiteningMatrix']);
%保存終了

return;

