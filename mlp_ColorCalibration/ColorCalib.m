
clear all; close all;
%----------------------------------------------------------------------
%                   　　画像の読み込み
%----------------------------------------------------------------------
ptn=imread('ColorMap.bmp');
figure(1); imshow(ptn);
gray_ptn = edge(rgb2gray(ptn),'sobel',0.05);
gray_ptn2 = imfill(gray_ptn, 'holes');
gray_ptn3 = medfilt2(gray_ptn2);
gray_ptn4 = medfilt2(gray_ptn3);
figure(2); imshow(gray_ptn4);
L1 = bwlabel(gray_ptn4, 8);
figure(3); axis image; imagesc(L1);

%----------------------------------------------------------------------
%                   　　　　　パラメータ
%----------------------------------------------------------------------
% ニューラルネットワークの構成
NUM_INPUT  = 3 ;     % 入力層の次元数
NUM_OUTPUT = 3 ;     % 出力層の次元数
NUM_LEARN  = 3000;   % 学習回数
NUM_HIDDEN = 20;     % 中間素子数
epsilon    = 0.3 ;   % 学習係数
 
% 結合荷重
w1=randn(NUM_INPUT+1, NUM_HIDDEN); % 入力層→中間層　-1～1の値　正規分布で乱数
w2=randn(NUM_HIDDEN, NUM_OUTPUT);  % 中間層→出力層


% 入力データ  %教師データ　--------------------------------------------------
% 縦：データ数，横：データの次元
% load RGBMAP;
X=[];
T=[];
for ll=1:1:8
    %赤 %黄 %緑 %マゼンタ %青 %シアン %黒 %白
    [r, c] = find(L1==ll); rc_R = [r c];
    pos=mean(rc_R);
    x1 = double(ptn( round(pos(1)), round(pos(2)), 1 ))/255.0;
    x2 = double(ptn( round(pos(1)), round(pos(2)), 2 ))/255.0;
    x3 = double(ptn( round(pos(1)), round(pos(2)), 3 ))/255.0;
    clear rc_R; clear r; clear c;
    X=[X; x1 x2 x3];
end

T=[1.0, 0.0, 0.0;
   1.0, 1.0, 0.0;
   0.0, 1.0, 0.0;
   1.0, 0.0, 1.0;
   0.0, 0.0, 1.0;
   0.0, 1.0, 1.0;
   0.0, 0.0, 0.0;
   1.0, 1.0, 1.0];

% 学習結果があれば，ロードし
% 学習をスキップする。
% load NEURO;
   

%----------------------------------------------------------------------
%     ニューラルネットワークの学習: 3層パーセプトロン　BP法
%----------------------------------------------------------------------
for l=1:1:NUM_LEARN     % 学習回数分
    for h=1:1:size(X,1) % データ数分
            
            % 中間層素子値の計算
            for j=1:1:NUM_HIDDEN-1
                X(h,NUM_INPUT+1)=1; % threshold
                S(j) = 0 ;
                for i=1:1:NUM_INPUT+1
                    S(j)=S(j)+w1(i,j)*X(h,i) ;
                end
                H(j) = 1.0/(1.0+exp(-1*S(j))) ;
            end
            % 出力素子の計算
            for k=1:1:NUM_OUTPUT
                H(NUM_HIDDEN) = 1 ;
                R(k) = 0 ;
                for j=1:1:NUM_HIDDEN
                    R(k) = R(k) + w2(j,k)*H(j);
                end
                Y(k) = 1.0/(1.0+exp( -1*R(k) )) ;
            end
            % 誤差の評価
            error(h) = 0 ;
            for k=1:1:NUM_OUTPUT
                error(h) = error(h) + ( (T(h,k)-Y(k)) * (T(h,k)-Y(k)) / 2.0 ) ;
            end
            % 逆方向動作
            % 出力層の逆伝播
            for k=1:1:NUM_OUTPUT
                Y_back(k) = (Y(k)-T(h,k))*(1-Y(k))*Y(k) ;
            end
            % 中間層の逆伝播
            for j=1:1:NUM_HIDDEN
                Q(j) = 0 ;
                for k=1:1:NUM_OUTPUT
                        Q(j) = Q(j) + w2(j,k)*Y_back(k)     ;
                end
                H_back(j) = Q(j) * ( 1 - H(j) ) * H(j) ;
            end
            % 重みの修正
            for i=1:1:NUM_INPUT+1
                for j=1:1:NUM_HIDDEN
        	        w1(i,j) = w1(i,j) - epsilon * X(h,i) * H_back(j);
                end
            end
            for j=1:1:NUM_HIDDEN
                for k=1:1:NUM_OUTPUT
        	        w2(j,k) = w2(j,k) - epsilon * H(j) * Y_back(k);
                end
            end
        % エラーの表示
        if mod(l,100) == 0
            Err = 0;
            for i=1:1:size(X,1)
                Err=Err+error(i);
                Err=Err/size(X,1);
            end
            Err
        end
    end        
end
    
    
% % ----------------------------------------------------------------------
% %     ニューラルネットワークの学習:  訓練データに対しての評価              
% % ----------------------------------------------------------------------
% C=X;  
% for h=1:1:size(C,1) % データ数分
%     % 中間層素子値の計算
%     for j=1:1:NUM_HIDDEN-1
%         C(h,NUM_INPUT+1)=1; % threshold
%         S(j) = 0 ;
%         for i=1:1:NUM_INPUT+1
%             S(j)=S(j)+w1(i,j)*C(h,i) ;
%         end
%         H(j) = 1.0/(1.0+exp(-1*S(j))) ;
%     end
% 
%     % 出力素子の計算
%     for k=1:1:NUM_OUTPUT
%         H(NUM_HIDDEN) = 1 ;
%         R(k) = 0 ;
%         for j=1:1:NUM_HIDDEN
%             R(k) = R(k) + w2(j,k)*H(j);
%         end
%         Y(k) = 1.0/(1.0+exp( -1*R(k) )) ;
%         OutImg(h,k)=Y(k);
%     end
% end
% 
% % 評価結果
% temp=sqrt(sum(( (255*(T-OutImg)).^2 )'));
% score=mean(temp')


% ----------------------------------------------------------------------
%     ニューラルネットワークの学習:  テストデータに対しての評価              
% ----------------------------------------------------------------------
clear C;
C=[];
ptntest=imread('ColorMapTest.bmp');

figure(100);
imshow(ptntest);
for col=1:1:size(ptntest,1)
    for row=1:1:size(ptntest,2)
        x1 = double(ptntest( col, row, 1 ))/255.0;
        x2 = double(ptntest( col, row, 2 ))/255.0;
        x3 = double(ptntest( col, row, 3 ))/255.0;
        C=[C; x1 x2 x3];
    end
end
OutImg=zeros(size(ptntest,1)*size(ptntest,2), 3);

for h=1:1:size(C,1)%データ数分
    %中間層素子値の計算
    for j=1:1:NUM_HIDDEN-1
        C(h,NUM_INPUT+1)=1;%threshold
        S(j) = 0 ;
        for i=1:1:NUM_INPUT+1
            S(j)=S(j)+w1(i,j)*C(h,i) ;
        end
        H(j) = 1.0/(1.0+exp(-1*S(j))) ;
    end

    %出力素子の計算
    for k=1:1:NUM_OUTPUT
        H(NUM_HIDDEN) = 1 ;
        R(k) = 0 ;
        for j=1:1:NUM_HIDDEN
            R(k) = R(k) + w2(j,k)*H(j);
        end
        Y(k) = 1.0/(1.0+exp( -1*R(k) )) ;
        OutImg(h,k)=Y(k);
    end
end

for h=1:1:size(C,1)%データ数分
    i=floor(h/size(ptntest,2))+1;
    j=mod(h,size(ptntest,2));
    if j==0; j=size(ptntest,2); end
    for k=1:1:NUM_OUTPUT
        Img(i,j,k)= OutImg(h,k);
    end
end

% NNで推定した結果を表示
figure(101); imshow(Img);

% 学習結果を保存
save('NEURO','w1','w2');
