
% MNIST�̐���������摜�𒊏o����
function ica_mnist

    % --------------------------------------------------------- �摜�̓Ǎ���
    % MNIST������͉摜�̓Ǎ���
    MnistImg   = loadMNISTImages('train-images.idx3-ubyte');
    % Mnistlabel = loadMNISTLabels('train-labels.idx1-ubyte');
    NumTrainingData = 20000;
    x = zeros( 256, NumTrainingData, 'double');
    
    % 15%�̊m���œ��͂Ƀ��C�_���m�C�Y������B�m�C�Y����ꂽ�����ǂ��炵���B
    % ���̘_�����Q�l
    for i=1:NumTrainingData
        if rand() > 0.15
            x(:, i) = MNIST(MnistImg(:, randi(size(MnistImg, 2)))); 
        else
            x(:, i) = randn(256, 1);
        end
    end
    
    covarianceMatrix = x*x'/size(x,2); % ���U�����U�s������߂�B
    [E, D] = eig(covarianceMatrix);    % �ŗL�x�N�g��E�A�ŗL�lD�����߂�B
    [dummy,order] = sort(diag(-D));
    rdim=32;                           % ���̘_������35���x�X�g�炵��
    E = E(:,order(1:rdim));            % rdim���܂łƂ�B
    d = diag(D); 
    d = real(d.^(-0.5));               % real�ɂ���̂�-1�Ƀ��[�g�����邽��
    D = diag(d(order(1:rdim)));        % E�ɂ��킹���āArdim���̐��܂łƂ�B
    x = D*E'*x;                        % ���F���s���������   
    
    m=rdim;                 % ���M���̐� rdim�łȂ��A10�ł��ǂ�
    c=size(x,1);            % �W�{���i�S�摜�̃s�N�Z�X���jrdim�̐�
    n=size(x,2);            % �ϑ��̐��F��L�Ⴞ��NumTrainingData
    
    % PCA�łǂ̂��炢�݂������H���`�F�b�N����B
    viewer_random((E*D^(-1))*x);  % 3�������_���ɕ\��: 
                                  % 16*16�T�C�Y�Ƀ��T�C�Y���ĕ\��
        
    %------------------------------------------------------------ ICA�̊w�K    
    eta=0.1;           % �w�K�W�� 0.6
    B=randn(rdim , m); % �����s��̏����� 

   
    % �w�K�����F��������200�`300�O�オ�K��
    for i=1:300       
        
        % ���M���̐���
        y = B'*x;
        
        DeltaB = ( (eye(m)+phi(y)*y'/n)*(B') )';  % ���R���z�����Ɋw�K
        B = B + eta * DeltaB;                     % �����s��̍X�V
        
        % �������������ꍇ
        % B = B*real((B'*B)^(-0.5));
    end
   
    % ���摜�̊m�F
    viewer((E*D^(-1))*B)
    
end

function y = phi(x)
    % y=-(x.^3); %��l���z�̏ꍇ
    y=-tanh(x); %���K���z�̏ꍇ�F�D�K�E�X
    % Extend Inforax�@�F�ȉ��̏ꍇ�͒��s�������Ȃ��Ɣ��U����
    % y=-(x+diag(sign(cumulant(x')))*tanh(x));
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

% �摜��S���`�悷��
function viewer(x)
    n=size(x,2);
    figure(10);
    % �\��������摜�̏c�A���̃T�C�Y�����߂�
    v_row = mod(n,8);  
    if v_row == 0; 
        v_col = floor(n/8);   
    else
        v_col = floor(n/8)+1;    
    end
    for i=1:n
        y=reshape(x(:,i),16,16);
        subplot(v_col,8,i)
        imshow(y,[min(y(:)) max(y(:))])
        drawnow
    end
end

% �摜�������_����3���`�悷��
function viewer_random(x)
    n=size(x,2);
    for i=1:3
        y=reshape(x(:, randi(n)),16,16);
        figure(i); imshow(y,[min(y(:)) max(y(:))]);
    end
end

% MNIST 28*28�̉摜��32*32�ɂȂ�悤�ɔw�i�F������B
function[InputImg] = MNIST(MnistImg)
    for k=1:size(MnistImg,2);%60000
        Xk=MnistImg(:,k);
        Xk=reshape(Xk, 28, 28);    
        base=zeros(32, 32);
        base(3:30,3:30)=double(Xk);        
        ImgG=double(base);
        [InputImg(:,k)] = DownSampling_32_32_to_16_16(32, 32, ImgG);
    end
end

% 32*32��16*16�Ƀ_�E���T���v�����O����, �����͂����m�N���摜
function[imgout] = DownSampling_32_32_to_16_16(hight, width, imgin)

    O_img = 255*ones(hight, width, 'double');
    O_img = double( imgin(:,:,1) );
 
    ptn = zeros(16,16);   
    dsh=hight/16;
    dsw=width/16;
    
    if dsh==dsw, ds=floor(dsh);
    else fprintf('Err');
    end
        
    l=1;
    for i=1:ds:hight
        k=1;
        for j=1:ds:width    
            ptn(l,k) = O_img(i,j);
            k=k+1;
        end
        l=l+1;
    end
	
	% �f�o�b�O�\��
    %figure(1);  colormap(gray);
    %axis image; imagesc(ptn);

    % ���K��
    I = double(ptn);
    I = I-mean(mean(I));         %�S��f���畽�ϒl������
    I = I/sqrt(mean(mean(I.^2)));%���U��1�ɂȂ�悤�ɐ��K������:�K�����f�[�^
    
    % �o��
    imgout= reshape(I, 256, 1);
end

