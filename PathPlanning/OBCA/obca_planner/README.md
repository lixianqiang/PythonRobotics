需要修改freespace_planner 将它的全局路径发出来 在 FreespacePlannerNode::onTimer() 中    
    trajectory_pub_->publish(partial_trajectory_);
    =》
    trajectory_pub_->publish(trajectory_);

如果使用 launch 文件， 则需要修改 launch/obca_planner.launch.py 中的 vehicle_info_param_path
