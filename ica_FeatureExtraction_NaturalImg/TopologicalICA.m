
function [] = TopologicalICA

    clear all; close all;
    
    % �w�K�f�[�^�̍쐬�@�w�F�w�K�f�[�^�i���̓f�[�^�j
    global X;
    % �p�b�`�̐��F50000�C�p�b�`��1�ӂ�̃s�N�Z���T�C�Y�F16�CPCA�ō팸���l�����鎟�����F160
    [X, whiteningMatrix, dewhiteningMatrix] = GetNaturalImages( 50000, 16, 160 );
    
    % �ݒ�ϐ�
    p.xdim = 16; % columns in map
    p.ydim = 10; % rows in map
    
    % �ߖT�̈�̍쐬
    [NBNZ,NB] = GenerateNB( p );
    
    % �w�̗�x�N�g���̐��F50000
    N = size(X,2);
     
    % �����̎�
    rand('seed',1);%1�͎�

    %�@s�i�X�p�[�X�M���j = B' * X 
    B = randn(size(X,1), p.xdim*p.ydim);  % 160*160�̗����s��
    % ������
    B = B*real((B'*B)^(-0.5));
    
    % B�̗�x�N�g���̐��F160
    n = size(B,2);   
    
    % �J��Ԃ��v�Z��
    iter=0;
    % �v�Z�X�e�b�v���ݕ�
    stepsize = 0.1;
    
	% �ϐ��m��
    obj = [];
    objiter = [];
    
    % �����J��Ԃ�
    while 1  
        % Increment iteration counter
        iter = iter+1;  
        fprintf('(%d)',iter);
         
        % Calculate linear filter responses and their squares
        U = B'*X; Usq = U.^2;

        % Calculate local energies E
        for ind=1:n
          E(ind,:) = NB(ind,NBNZ{ind}) * Usq(NBNZ{ind},:);
        end

        % Take nonlinearity
        g = -((0.005 + E).^(-0.5));

        % Calculate convolution with neighborhood
        for ind=1:n
          F(ind,:) = NB(ind,NBNZ{ind}) * g(NBNZ{ind},:);
        end

        % This is the total gradient
        % dB = X*(U.*F)'/N; % ����
        dB = X*(U.*g)'/N;   % �����炪�^
        
        % ADAPT STEPSIZE FOR GRADIENT ALGORITHMS
        % Perform this adaptation only every 5 steps
        if rem(iter,5)==0 | iter==1

          % Take different length steps
          Bc{1} = B + 0.5*stepsize*dB;
          Bc{2} = B + 1.0*stepsize*dB;
          Bc{3} = B + 2.0*stepsize*dB;

          % Orthogonalize each one
          for i=1:3, Bc{i} = Bc{i}*real((Bc{i}'*Bc{i})^(-0.5)); end

          % Calculate objective values in each case
          for i=1:3
            Usq = (Bc{i}'*X).^2;
            for ind=1:n
                E(ind,:)= NB(ind,NBNZ{ind}) * Usq(NBNZ{ind},:);%E�̌v�Z 
            end
            objective(i) = mean(mean(sqrt(0.005+E)));
          end

          % Compare objective values, pick step giving minimum
          if objective(1)<objective(2) & objective(1)<objective(3)
            % Take shorter step�@����ɃX�e�b�v����������
            stepsize = stepsize/2;
            fprintf('Stepsize now: %.4f\n',stepsize);
            obj = [obj objective(1)];
          elseif objective(1)>objective(3) & objective(2)>objective(3)
            % Take longer step�@����ɃX�e�b�v����傫��
            stepsize = stepsize*2;
            fprintf('Stepsize now: %.4f\n',stepsize);
            obj = [obj objective(3)];
          else
            % Don't change step
            obj = [obj objective(2)];
          end

          objiter = [objiter iter];
          fprintf('\nObjective value: %.6f\n',obj(end));

        end % 5���1�x �X�e�b�v�T�C�Y��ς���

        % stepsize�����߂���A�����ōX�V
        B = B + stepsize*dB;

        % Ortogonalize : ������
        % (equal to decorrelation since we are in whitened space)
        B = B*real((B'*B)^(-0.5));
        
        % �m�F�������s��A�ȉ��̂Q��
        A = dewhiteningMatrix * B; % ���x�N�g��
        W = B' * whiteningMatrix;  % ���̂܂܂̉摜�f�[�^�����͂��ꂽ�Ƃ���W�ɂ�����s��A���F���������
            
%         % Write the results to disk
%         % rem ���Z��̏�] 5���1��̕p�x�ŏ�����
%         if rem(iter, 5)==0 | iter==1
%             fname='tica.mat';
%             fprintf(['Writing file: ' fname '...']);
%             % �ȉ��̕ϐ���MAT�t�@�C���ɕۑ�
%             eval(['save ' fname ' W A p iter obj objiter']);
%             fprintf(' Done!\n');      
%         end
      
        if iter == 100
            % 2�Ԗڂ̈����͉摜��\������ۂ̃T�C�Y�C���傫���\���������Ƃ���
            % �傫������
            % ���摜�̊m�F
            visual( A, 2, 16 ); 
            break;
        end
        
    end % while

        
end %���C���֐��I��


% GenerateNB - generates the neighborhood matrix
% �ߖT�Z���̓��o
function[NBNZ,NB] = GenerateNB( p )

    % This will hold the neighborhood function entries
    NB = zeros(p.xdim*p.ydim*[1 1]);

    % Step through nodes one at a time to build the matrix
    ind = 0;
    for y=1:p.ydim
      for x=1:p.xdim
          
        ind = ind+1;
        
        % Rectangular neighbors
        [xn,yn] = meshgrid( (x-1):(x+1), (y-1):(y+1) );
        xn = reshape(xn,[1 9]);
        yn = reshape(yn,[1 9]);

        % ���E�����̏���
        i = find(yn<1); yn(i)=yn(i)+p.ydim;
        i = find(yn>p.ydim); yn(i)=yn(i)-p.ydim;
        i = find(xn<1); xn(i)=xn(i)+p.xdim;
        i = find(xn>p.xdim); xn(i)=xn(i)-p.xdim;

        % Set neighborhood
        % 160*160�̃}�g���b�N�X�ŁC�ߖT�̈�݂̂P
        NB( ind, (yn-1)*p.xdim + xn )=1;
      end
    end

    % For each unit, calculate the non-zero columns!
    % �ߖT�Z���̑g�ݍ��킹��NBNZ�ɑ��
    for i=1:p.xdim*p.ydim
        % ��[���̂h�m�c�d�w��������A�ߖT�̈�̏ꏊ�̂�
        NBNZ{i} = find(NB(i,:));
    end
end

% �R�X�g�֐�
function y = phi(x)
    % �D�K�E�X���z
    y=-1.0*tanh(x);
end