
% ３つの画像を混合したものを分離する。
function ica_img

    clear all; close all;
    
    % --------------------------------------------------------- 画像の読込み
    % 画像その1：岩
    imgin = imread('pic01.jpg');   [s1(:,:,1)] = DownSampling(180, 270, imgin);
    % 画像その2：花    
    imgin = imread('pic02.jpg');   [s2(:,:,1)] = DownSampling(180, 270, imgin);
    % 画像その3：ねこちゃん 
    imgin = imread('pic03.jpg');   [s3(:,:,1)] = DownSampling(180, 270, imgin);
    % 画像その4：ドットパターン
    %imgin = imread('0.jpg');       [s4(:,:,1)] = DownSampling(600, 600, imgin);
    
    % 混合数によって以下を変える。
    %s=[s1(:) s2(:) ]';             % 源信号数が2つの場合
    s=[s1(:) s2(:) s3(:)]';         % 源信号数が3つの場合
    %s=[s1(:) s2(:) s3(:) s4(:)]';  % 源信号数4つの場合    
    
    m=size(s,1); % 源信号の数
    c=3;       % 観測信号の数
    n=size(s,2); % 標本数（全画像のピクセス数）128*128にリサイズ
    
    % --------------------------------------------------------- 混合信号作成
    % 観測信号の設定
    A=randn(c, m); % A=eye(3);%単位行列でも行けること確認済
    x=A*s;      % 混合処理
    x=nor(x);   % 平均0，分散1に正規化(内部関数)
   
    % 重ね合わさった画像の確認
    viewer(x); pause   % 全部画像を表示
    % viewer_random(x);  % 3枚ランダムに表示
        

    %------------------------------------------------------------ ICAの学習    
    eta=0.6;     % 学習係数 0.6
    W=eye(m, c); % 分離行列の初期化 

    % 学習反復：反復数は200～300前後が適切
    for i=1:300
        y = W*x;
        
        % 評価関数、以下から選択する。
        %DeltaW = (W’)^(-1) + phi(y)*x’/n;    %最急勾配方向に学習
        %DeltaW = ((phi(y)*y'-y*phi(y)')/n)*W;  %直交の制約を入れた場合
        DeltaW = (eye(m)+phi(y)*y'/n)*W;        %自然勾配方向に学習
                                                %正方行列,正方行列でない場合共にOK
        
        % 学習係数を進行に合わせて小さくする。
        if i<100
          eta=0.6;
        else
          eta=0.2;  
        end
        
        % 分離行列の更新
        W = W + eta * DeltaW;
        
        % 直交化処理はやらない方が精度が安定する。
        % ただし、発散に気を付けること。
    end
    
    % 分離画像の確認
    viewer(y)
        
end

function y = phi(x)
    %y=-(x.^3); %一様分布の場合
    %y=-tanh(x); %正規分布の場合：優ガウス
    %Extend Inforax法：以下
    y=-(x+diag(sign(cumulant(x')))*tanh(x));
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

% 128*128のイメージサイズにダウンサンプリングする
function[imgout] = DownSampling(hight, width, imgin)

    R_img = 255*ones(hight, width, 'double'); 
    G_img = 255*ones(hight, width, 'double');  
    B_img = 255*ones(hight, width, 'double');
    R_img = double( imgin(:,:,1) );         
    G_img = double( imgin(:,:,2) );          
    B_img = double( imgin(:,:,3) );
    
    % 出力用変数
    ptn = zeros(128,128);
    
    % スキップ数
    dsh=floor(hight/128);
    dsw=floor(width/128);
    
    % サンプリング処理
    l=1;
    for i=1:dsh:128*dsh
        k=1;
        for j=1:dsw:128*dsw    
            ptn(l,k) = 0.114*B_img(i,j) + 0.587*G_img(i,j) + 0.299*R_img(i,j);
            k=k+1;
        end
        l=l+1;
    end
        
    % 出力
    imgout= ptn;
    
end

% 画像を全部描画する
function viewer(x)
    n=size(x,1);
    figure(10);
    for i=1:n
        y=reshape(x(i,:),128,128);
        subplot(1,n,i)
        % Image processing toolbox がない場合，image() を使う．
        imshow(y,[min(y(:)) max(y(:))])
        drawnow
    end
end

% 画像をランダムに3枚描画する
function viewer_random(x)
    n=size(x,1);
    for i=1:3
        y=reshape(x(randi(n), :),128,128);
        figure(i); imshow(y,[min(y(:)) max(y(:))]);
    end
end
