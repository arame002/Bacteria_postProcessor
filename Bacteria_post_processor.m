% bacteria post processor

nbacteria = 50 ;
domainSize = 1000 ;

b_trajectoriesX = [] ;
b_trajectoriesY = [] ;
b_locVelocity = [] ;
b_locFriction = [] ;
b_Orientation = [] ;
b_chemical = [] ;

sourcesX = [] ;
sourcesY = [] ;

nReverse = [] ;
time_toSource = [] ;
time_inSource = [] ;
init_maxRunDuration = [] ;

vDrift = [];

R_threshold = 10 ;
t_in = [] ;
t_to = [] ;



for fileNum = 1:2000
    disp(fileNum)

    filename = ['Bacteria_' num2str(fileNum) '.txt'];
    fileID = fopen(filename);
    C = textscan(fileID,'%f %f %f %f %f %f','Delimiter','\t');
    fclose(fileID);
    b_trajectoriesX = [b_trajectoriesX ; transpose( C{1}) ] ;
    b_trajectoriesY = [b_trajectoriesY ; transpose( C{2}) ] ;
    b_locVelocity   = [b_locVelocity   ; transpose( C{3}) ] ;
    b_locFriction   = [b_locFriction   ; transpose( C{4}) ] ;
    b_Orientation   = [b_Orientation   ; transpose( C{5}) ] ;
    b_chemical      = [b_chemical      ; transpose( C{6}) ] ;
    
end
fileName = 'sourceLocation.txt' ;
fileID = fopen(fileName);
C = textscan(fileID,'%f %f','Delimiter','\t');
fclose(fileID);
sourcesX = C{1} ;
sourcesY = C{2} ;

fileName = 'HistogramReversal.txt' ;
fileID = fopen(fileName);
C = textscan(fileID,'%f %f %f %f %f %f','Delimiter','\t');
fclose(fileID);

nReverse = C{2} ;
time_toSource= C{3} ;
time_inSource = C{4} ;
init_maxRunDuration = C{6} ;

vx = [b_trajectoriesX]- b_trajectoriesX(1,:) ; 
vy = [b_trajectoriesY]- b_trajectoriesY(1,:) ;
dist = [sourcesX - b_trajectoriesX(1,:) ; sourcesY - b_trajectoriesY(1,:)];
xDrift = (vx .* dist(1,:) + vy .* dist(2,:) ) ./ sqrt(dist(1,:) .* dist(1,:) + dist(2,:) .* dist(2,:) );
vDrift = xDrift(2:end,:) ./  find(xDrift(:,1)) ;

distToSourceX = [b_trajectoriesX] - sourcesX ;
distToSourceY = [b_trajectoriesY] - sourcesY ;
distToSource = ( distToSourceX .* distToSourceX + distToSourceY .* distToSourceY ).^ 0.5 ;
xDrift2 = distToSource(1,:) - [distToSource] ;
vDrift2 = xDrift2(2:end,:) ./  find(xDrift2(:,1)) ;

vdAvg = [mean(vDrift2,2), std(vDrift2,0,2)] ;
distToSourceAvg = [mean(distToSource,2), std(distToSource,0,2)] ;



dist2D = [sqrt((sourcesX - b_trajectoriesX).^2 + (sourcesY - b_trajectoriesY).^2 ) ] ;
Row = [] ;
Col = [] ;
vd = [] ;
vd2 = [] ;
for i = 1:size(dist2D,2)
    [row,col] = find(dist2D(:,i)< R_threshold, 1);
    Row = [Row , row ];
    Col = [Col, col* i] ;
    vd = [vd , vDrift(row,col)];
    vd2 = [vd2 , vDrift2(row,col)];
    
end
t_to = [Col;Row;vd;vd2];
%VdAvg = mean(t_to(3,:)) ;

t_in = zeros(size(dist2D,1),size(dist2D,2));

for i = 1:size(dist2D,1)
    for j = 1:size(dist2D,2)
        
        if (dist2D(i,j)< R_threshold )

            t_in(i+1,j) = t_in(i,j)+ 0.1 ; 
        else
            t_in(i+1,j) = t_in(i,j) ;
            
        end
    
    end
end

tIn_avg = [mean(t_in,2)] ;
   









