
% MNISTの数字から基底画像を抽出する
function ica_mnist

    % --------------------------------------------------------- 画像の読込み
    % MNISTから入力画像の読込み
    MnistImg   = loadMNISTImages('train-images.idx3-ubyte');
    % Mnistlabel = loadMNISTLabels('train-labels.idx1-ubyte');
    NumTrainingData = 20000;
    x = zeros( 256, NumTrainingData, 'double');
    
    % 15%の確率で入力にライダムノイズを入れる。ノイズを入れた方が良いらしい。
    % 阪大の論文を参考
    for i=1:NumTrainingData
        if rand() > 0.15
            x(:, i) = MNIST(MnistImg(:, randi(size(MnistImg, 2)))); 
        else
            x(:, i) = randn(256, 1);
        end
    end
    
    covarianceMatrix = x*x'/size(x,2); % 分散共分散行列を求める。
    [E, D] = eig(covarianceMatrix);    % 固有ベクトルE、固有値Dを求める。
    [dummy,order] = sort(diag(-D));
    rdim=32;                           % 阪大の論文だと35がベストらしい
    E = E(:,order(1:rdim));            % rdim次までとる。
    d = diag(D); 
    d = real(d.^(-0.5));               % realにするのは-1にルートが入るため
    D = diag(d(order(1:rdim)));        % Eにあわせえて、rdim次の数までとる。
    x = D*E'*x;                        % 白色化行列をかける   
    
    m=rdim;                 % 源信号の数 rdimでなく、10でも良い
    c=size(x,1);            % 標本数（全画像のピクセス数）rdimの数
    n=size(x,2);            % 観測の数：上記例だとNumTrainingData
    
    % PCAでどのくらい鈍ったか？をチェックする。
    viewer_random((E*D^(-1))*x);  % 3枚ランダムに表示: 
                                  % 16*16サイズにリサイズして表示
        
    %------------------------------------------------------------ ICAの学習    
    eta=0.1;           % 学習係数 0.6
    B=randn(rdim , m); % 分離行列の初期化 

   
    % 学習反復：反復数は200～300前後が適切
    for i=1:300       
        
        % 源信号の推定
        y = B'*x;
        
        DeltaB = ( (eye(m)+phi(y)*y'/n)*(B') )';  % 自然勾配方向に学習
        B = B + eta * DeltaB;                     % 分離行列の更新
        
        % 直交処理入れる場合
        % B = B*real((B'*B)^(-0.5));
    end
   
    % 基底画像の確認
    viewer((E*D^(-1))*B)
    
end

function y = phi(x)
    % y=-(x.^3); %一様分布の場合
    y=-tanh(x); %正規分布の場合：優ガウス
    % Extend Inforax法：以下の場合は直行化もしないと発散する
    % y=-(x+diag(sign(cumulant(x')))*tanh(x));
end

% キュムラント
function y=cumulant(x)
    y=mean(x.^4)-3*mean(x.^2).^2;
end

% 信号y の平均を0，分散を1 に規格化する
function y=nor(y)
    y=y-repmat(mean(y,2),1,size(y,2));
    y=diag(mean(y.^2,2).^(-1/2))*y;
end

% 白色化
function [y,v,d]=prewhitening(x)
    x=x-repmat(mean(x,2),1,size(x,2));
    [v,d]=eig(x*x'/size(x,2));
    d=d^(-1/2);
    y=v*d*v'*x;
end

% 画像を全部描画する
function viewer(x)
    n=size(x,2);
    figure(10);
    % 表示する基底画像の縦、横のサイズを決める
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

% 画像をランダムに3枚描画する
function viewer_random(x)
    n=size(x,2);
    for i=1:3
        y=reshape(x(:, randi(n)),16,16);
        figure(i); imshow(y,[min(y(:)) max(y(:))]);
    end
end

% MNIST 28*28の画像を32*32になるように背景色を入れる。
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

% 32*32→16*16にダウンサンプリングする, ※入力がモノクロ画像
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
	
	% デバッグ表示
    %figure(1);  colormap(gray);
    %axis image; imagesc(ptn);

    % 正規化
    I = double(ptn);
    I = I-mean(mean(I));         %全画素から平均値を引く
    I = I/sqrt(mean(mean(I.^2)));%分散が1になるように正規化する:規準化データ
    
    % 出力
    imgout= reshape(I, 256, 1);
end

