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

% We have a total of 13 images, �摜�̐�:13��
dataNum = 13;

% This is how many patches to take per image,�@1�����炢���p�b�`���Ƃ邩�H
getsample = floor(samples/dataNum);

% Initialize the matrix to hold the patches, �p�b�`�摜�̏�����
X = zeros(winsize^2,getsample*dataNum);

% �؂�o���p�b�`���C�����l�P
sampleNum = 1; 

% �摜�������J��Ԃ�
for i=(1:dataNum)

  % Even things out (take enough from last image)
  if i==dataNum
      % �[���ɂȂ�Ȃ��悤��samples���ɐ��m�ɂȂ�悤�ɁA�Ō�̉摜�̂݃p�b�`�������
      getsample = samples-sampleNum+1;
  end
  
  % Load the image, �摜�̃��[�h
  I = imread([ num2str(i) '.tiff']);

  % Normalize to zero mean and unit variance
  % 1���̉摜�ɑ΂��āC�܂����K��������
  I = double(I);
  I = I-mean(mean(I));          % �S��f���畽�ϒl������
  I = I/sqrt(mean(mean(I.^2))); % ���U��1�ɂȂ�悤�ɐ��K������:�K�����f�[�^
  
  % Sample 
  fprintf('Sampling image %d...\n',i);
  sizex = size(I,2); sizey = size(I,1);
  posx = floor(rand(1,getsample)*(sizex-winsize-2))+1; % �����ɂāA�摜����p�b�`�̎擾�̏ꏊ�����߂�
  posy = floor(rand(1,getsample)*(sizey-winsize-1))+1; % �����ɂāA�摜����p�b�`�̎擾�̏ꏊ�����߂�
  
  for j=1:getsample
    % �c:�s�N�Z�����C256�F�P�̃p�b�`�f�[�^16*16�B
    % ���Ƀp�b�`���������B1���̉摜����getsample�擾���A�摜�����������B
    X(:,sampleNum) = reshape( I(posy(1,j):posy(1,j)+winsize-1, ...
			posx(1,j):posx(1,j)+winsize-1),[winsize^2 1]);
    sampleNum=sampleNum+1;
  end 
  
end

%----------------------------------------------------------------------
% Subtract local mean gray-scale value from each patch
%----------------------------------------------------------------------
fprintf('Subtracting local mean...\n');
% �e�p�b�`���Ƃɕ��ς��O�ɂȂ�悤�ɂ���
X = X-ones(size(X,1),1)*mean(X);

%----------------------------------------------------------------------
% Reduce the dimension and whiten at the same time!
%----------------------------------------------------------------------

% Calculate the eigenvalues and eigenvectors of covariance matrix.
fprintf ('Calculating covariance...\n');
% ���U�����U�s������߂�B
covarianceMatrix = X*X'/size(X,2);
% �ŗL�x�N�g��E�A�ŗL�lD�����߂�B
[E, D] = eig(covarianceMatrix);

% Sort the eigenvalues and select subset, and whiten
fprintf('Reducing dimensionality and whitening...\n');
[dummy,order] = sort(diag(-D)); % D���傫���̂���ɗ���悤�Ƀ\�[�g����B
E = E(:,order(1:rdim));         % rdim(160)�̐��܂łƂ�B��^�����傫��������P�U�O��
d = diag(D); 
d = real(d.^(-0.5));            % real�ɂ���̂�-1�Ƀ��[�g�����邽��
D = diag(d(order(1:rdim)));     % E�ɂ��킹���āArdim(160)�̐��܂łƂ�B
X = D*E'*X;                     % ���F���s���������

whiteningMatrix = D*E';
dewhiteningMatrix = E*D^(-1);

%�ۑ��F
fname='whitening.mat';
fprintf(['Writing file: ' fname '...']);
eval(['save ' fname ' whiteningMatrix dewhiteningMatrix']);
%�ۑ��I��

return;

