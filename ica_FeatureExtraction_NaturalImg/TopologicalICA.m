
function [] = TopologicalICA

    clear all; close all;
    
    % 学習データの作成　Ｘ：学習データ（入力データ）
    global X;
    % パッチの数：50000，パッチの1辺りのピクセルサイズ：16，PCAで削減し考慮する次元数：160
    [X, whiteningMatrix, dewhiteningMatrix] = GetNaturalImages( 50000, 16, 160 );
    
    % 設定変数
    p.xdim = 16; % columns in map
    p.ydim = 10; % rows in map
    
    % 近傍領域の作成
    [NBNZ,NB] = GenerateNB( p );
    
    % Ｘの列ベクトルの数：50000
    N = size(X,2);
     
    % 乱数の種
    rand('seed',1);%1は種

    %　s（スパース信号） = B' * X 
    B = randn(size(X,1), p.xdim*p.ydim);  % 160*160の乱数行列
    % 直交化
    B = B*real((B'*B)^(-0.5));
    
    % Bの列ベクトルの数：160
    n = size(B,2);   
    
    % 繰り返し計算回数
    iter=0;
    % 計算ステップ刻み幅
    stepsize = 0.1;
    
	% 変数確保
    obj = [];
    objiter = [];
    
    % 無限繰り返し
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
        % dB = X*(U.*F)'/N; % 原文
        dB = X*(U.*g)'/N;   % こちらが真
        
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
                E(ind,:)= NB(ind,NBNZ{ind}) * Usq(NBNZ{ind},:);%Eの計算 
            end
            objective(i) = mean(mean(sqrt(0.005+E)));
          end

          % Compare objective values, pick step giving minimum
          if objective(1)<objective(2) & objective(1)<objective(3)
            % Take shorter step　さらにステップ幅を小さく
            stepsize = stepsize/2;
            fprintf('Stepsize now: %.4f\n',stepsize);
            obj = [obj objective(1)];
          elseif objective(1)>objective(3) & objective(2)>objective(3)
            % Take longer step　さらにステップ幅を大きく
            stepsize = stepsize*2;
            fprintf('Stepsize now: %.4f\n',stepsize);
            obj = [obj objective(3)];
          else
            % Don't change step
            obj = [obj objective(2)];
          end

          objiter = [objiter iter];
          fprintf('\nObjective value: %.6f\n',obj(end));

        end % 5回に1度 ステップサイズを変える

        % stepsizeを決めた後、ここで更新
        B = B + stepsize*dB;

        % Ortogonalize : 直交化
        % (equal to decorrelation since we are in whitened space)
        B = B*real((B'*B)^(-0.5));
        
        % 確認したい行列、以下の２つ
        A = dewhiteningMatrix * B; % 基底ベクトル
        W = B' * whiteningMatrix;  % そのままの画像データが入力されたときにWにあたる行列、白色化も入れる
            
%         % Write the results to disk
%         % rem 除算後の剰余 5回に1回の頻度で書込み
%         if rem(iter, 5)==0 | iter==1
%             fname='tica.mat';
%             fprintf(['Writing file: ' fname '...']);
%             % 以下の変数をMATファイルに保存
%             eval(['save ' fname ' W A p iter obj objiter']);
%             fprintf(' Done!\n');      
%         end
      
        if iter == 100
            % 2番目の引数は画像を表示する際のサイズ，より大きく表示したいときは
            % 大きくする
            % 基底画像の確認
            visual( A, 2, 16 ); 
            break;
        end
        
    end % while

        
end %メイン関数終了


% GenerateNB - generates the neighborhood matrix
% 近傍セルの導出
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

        % 境界部分の処理
        i = find(yn<1); yn(i)=yn(i)+p.ydim;
        i = find(yn>p.ydim); yn(i)=yn(i)-p.ydim;
        i = find(xn<1); xn(i)=xn(i)+p.xdim;
        i = find(xn>p.xdim); xn(i)=xn(i)-p.xdim;

        % Set neighborhood
        % 160*160のマトリックスで，近傍領域のみ１
        NB( ind, (yn-1)*p.xdim + xn )=1;
      end
    end

    % For each unit, calculate the non-zero columns!
    % 近傍セルの組み合わせをNBNZに代入
    for i=1:p.xdim*p.ydim
        % 非ゼロのＩＮＤＥＸを見つける、近傍領域の場所のみ
        NBNZ{i} = find(NB(i,:));
    end
end

% コスト関数
function y = phi(x)
    % 優ガウス分布
    y=-1.0*tanh(x);
end