 index = 6;
 eval(['b_trajectoriesX_' num2str(index)  '=b_trajectoriesX'])  ;
 eval(['b_trajectoriesY_' num2str(index)  '=b_trajectoriesY'])  ;
 eval(['b_chemical_' num2str(index)  '=b_chemical'])  ;
 eval(['b_locFriction_' num2str(index)  '=b_locFriction'])  ;
 eval(['b_locVelocity_' num2str(index)  '=b_locVelocity'])  ;
 eval(['b_Orientation_' num2str(index)  '=b_Orientation'])  ;
 eval(['nReverse_' num2str(index)  '=nReverse'])  ;

 eval(['t_in_' num2str(index)  '=t_in'])  ;
 eval(['t_to_' num2str(index)  '=t_to'])  ;
 eval(['time_inSource_' num2str(index)  '=time_inSource'])  ;
 eval(['time_toSource_' num2str(index)  '=time_toSource'])  ;
 eval(['init_maxRunDuration_' num2str(index)  '=init_maxRunDuration'])  ;
 eval(['vDrift1_' num2str(index)  '=vDrift'])  ;
 eval(['vDrift2_' num2str(index)  '=vDrift2'])  ;
 
 eval(['vdAvg_' num2str(index)  '=vdAvg'])  ;
 eval(['distToSourceAvg_' num2str(index)  '=distToSourceAvg'])  ;
 eval(['tIn_avg_' num2str(index)  '=tIn_avg'])  ;




%figure(1);
%plotFrameNumber = 1500 ;
%hold on ;
%bar(histogramX1,histogramY1(plotFrameNumber,:),'FaceColor','b','EdgeColor','b');
%bar(histogramX2,histogramY2(plotFrameNumber,:),'FaceColor','r','EdgeColor','r');
%bar(histogramX3,histogramY3(plotFrameNumber,:),'FaceColor','g','EdgeColor','g');
%bar(histogramX4,histogramY4(plotFrameNumber,:),'FaceColor','y','EdgeColor','y');
%plot(tissueHeight3,'LineWidth',5,'Color','g');
%axis( [-10 100 0 200])
%ylabel('# Bacteria')
%xlabel('x location')
%set(gca,'LineWidth',2,'FontSize',25) 
%hold off
%legend('UniformProduction')


%save 'longWrap50_Nov11.mat'

