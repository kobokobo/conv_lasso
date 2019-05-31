
% �R�̉摜�������������̂𕪗�����B
function ica_img

    clear all; close all;
    
    % --------------------------------------------------------- �摜�̓Ǎ���
    % �摜����1�F��
    imgin = imread('pic01.jpg');   [s1(:,:,1)] = DownSampling(180, 270, imgin);
    % �摜����2�F��    
    imgin = imread('pic02.jpg');   [s2(:,:,1)] = DownSampling(180, 270, imgin);
    % �摜����3�F�˂������ 
    imgin = imread('pic03.jpg');   [s3(:,:,1)] = DownSampling(180, 270, imgin);
    % �摜����4�F�h�b�g�p�^�[��
    %imgin = imread('0.jpg');       [s4(:,:,1)] = DownSampling(600, 600, imgin);
    
    % �������ɂ���Ĉȉ���ς���B
    %s=[s1(:) s2(:) ]';             % ���M������2�̏ꍇ
    s=[s1(:) s2(:) s3(:)]';         % ���M������3�̏ꍇ
    %s=[s1(:) s2(:) s3(:) s4(:)]';  % ���M����4�̏ꍇ    
    
    m=size(s,1); % ���M���̐�
    c=3;       % �ϑ��M���̐�
    n=size(s,2); % �W�{���i�S�摜�̃s�N�Z�X���j128*128�Ƀ��T�C�Y
    
    % --------------------------------------------------------- �����M���쐬
    % �ϑ��M���̐ݒ�
    A=randn(c, m); % A=eye(3);%�P�ʍs��ł��s���邱�Ɗm�F��
    x=A*s;      % ��������
    x=nor(x);   % ����0�C���U1�ɐ��K��(�����֐�)
   
    % �d�ˍ��킳�����摜�̊m�F
    viewer(x); pause   % �S���摜��\��
    % viewer_random(x);  % 3�������_���ɕ\��
        

    %------------------------------------------------------------ ICA�̊w�K    
    eta=0.6;     % �w�K�W�� 0.6
    W=eye(m, c); % �����s��̏����� 

    % �w�K�����F��������200�`300�O�オ�K��
    for i=1:300
        y = W*x;
        
        % �]���֐��A�ȉ�����I������B
        %DeltaW = (W�f)^(-1) + phi(y)*x�f/n;    %�ŋ}���z�����Ɋw�K
        %DeltaW = ((phi(y)*y'-y*phi(y)')/n)*W;  %�����̐������ꂽ�ꍇ
        DeltaW = (eye(m)+phi(y)*y'/n)*W;        %���R���z�����Ɋw�K
                                                %�����s��,�����s��łȂ��ꍇ����OK
        
        % �w�K�W����i�s�ɍ��킹�ď���������B
        if i<100
          eta=0.6;
        else
          eta=0.2;  
        end
        
        % �����s��̍X�V
        W = W + eta * DeltaW;
        
        % �����������͂��Ȃ��������x�����肷��B
        % �������A���U�ɋC��t���邱�ƁB
    end
    
    % �����摜�̊m�F
    viewer(y)
        
end

function y = phi(x)
    %y=-(x.^3); %��l���z�̏ꍇ
    %y=-tanh(x); %���K���z�̏ꍇ�F�D�K�E�X
    %Extend Inforax�@�F�ȉ�
    y=-(x+diag(sign(cumulant(x')))*tanh(x));
end

% �L���������g
function y=cumulant(x)
    y=mean(x.^4)-3*mean(x.^2).^2;
end

% �M��y �̕��ς�0�C���U��1 �ɋK�i������
function y=nor(y)
    y=y-repmat(mean(y,2),1,size(y,2));
    y=diag(mean(y.^2,2).^(-1/2))*y;
end

% ���F��
function [y,v,d]=prewhitening(x)
    x=x-repmat(mean(x,2),1,size(x,2));
    [v,d]=eig(x*x'/size(x,2));
    d=d^(-1/2);
    y=v*d*v'*x;
end

% 128*128�̃C���[�W�T�C�Y�Ƀ_�E���T���v�����O����
function[imgout] = DownSampling(hight, width, imgin)

    R_img = 255*ones(hight, width, 'double'); 
    G_img = 255*ones(hight, width, 'double');  
    B_img = 255*ones(hight, width, 'double');
    R_img = double( imgin(:,:,1) );         
    G_img = double( imgin(:,:,2) );          
    B_img = double( imgin(:,:,3) );
    
    % �o�͗p�ϐ�
    ptn = zeros(128,128);
    
    % �X�L�b�v��
    dsh=floor(hight/128);
    dsw=floor(width/128);
    
    % �T���v�����O����
    l=1;
    for i=1:dsh:128*dsh
        k=1;
        for j=1:dsw:128*dsw    
            ptn(l,k) = 0.114*B_img(i,j) + 0.587*G_img(i,j) + 0.299*R_img(i,j);
            k=k+1;
        end
        l=l+1;
    end
        
    % �o��
    imgout= ptn;
    
end

% �摜��S���`�悷��
function viewer(x)
    n=size(x,1);
    figure(10);
    for i=1:n
        y=reshape(x(i,:),128,128);
        subplot(1,n,i)
        % Image processing toolbox ���Ȃ��ꍇ�Cimage() ���g���D
        imshow(y,[min(y(:)) max(y(:))])
        drawnow
    end
end

% �摜�������_����3���`�悷��
function viewer_random(x)
    n=size(x,1);
    for i=1:3
        y=reshape(x(randi(n), :),128,128);
        figure(i); imshow(y,[min(y(:)) max(y(:))]);
    end
end
