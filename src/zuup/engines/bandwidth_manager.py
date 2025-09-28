"""Enhanced bandwidth management with fair allocation algorithms."""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AllocationAlgorithm(Enum):
    """Bandwidth allocation algorithms."""
    
    EQUAL = "equal"  # Equal allocation among all workers
    PROPORTIONAL = "proportional"  # Proportional to current speed
    WEIGHTED_FAIR = "weighted_fair"  # Weighted fair queuing
    PRIORITY_BASED = "priority_based"  # Priority-based allocation
    ADAPTIVE = "adaptive"  # Adaptive based on performance


class WorkerPriority(Enum):
    """Worker priority levels."""
    
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class BandwidthAllocation:
    """Represents a bandwidth allocation for a worker."""
    
    def __init__(
        self,
        worker_id: str,
        allocated_bandwidth: int,
        priority: WorkerPriority = WorkerPriority.NORMAL,
        weight: float = 1.0,
    ) -> None:
        """
        Initialize bandwidth allocation.
        
        Args:
            worker_id: Worker identifier
            allocated_bandwidth: Allocated bandwidth in bytes per second
            priority: Worker priority level
            weight: Worker weight for allocation algorithms
        """
        self.worker_id = worker_id
        self.allocated_bandwidth = allocated_bandwidth
        self.priority = priority
        self.weight = weight
        self.last_updated = time.time()
        
        # Usage tracking
        self.actual_usage = 0
        self.usage_history: deque[tuple[float, int]] = deque(maxlen=60)  # 1 minute of history
        
    def update_usage(self, current_usage: int) -> None:
        """Update current usage statistics."""
        self.actual_usage = current_usage
        self.usage_history.append((time.time(), current_usage))
        
    def get_average_usage(self, window_seconds: int = 30) -> float:
        """Get average usage over a time window."""
        if not self.usage_history:
            return 0.0
            
        cutoff_time = time.time() - window_seconds
        recent_usage = [usage for timestamp, usage in self.usage_history if timestamp >= cutoff_time]
        
        return sum(recent_usage) / len(recent_usage) if recent_usage else 0.0
        
    def get_utilization_rate(self) -> float:
        """Get current utilization rate (0.0 to 1.0)."""
        if self.allocated_bandwidth == 0:
            return 0.0
        return min(1.0, self.actual_usage / self.allocated_bandwidth)


class EnhancedBandwidthManager:
    """Enhanced bandwidth manager with multiple allocation algorithms."""
    
    def __init__(
        self,
        total_bandwidth: Optional[int] = None,
        algorithm: AllocationAlgorithm = AllocationAlgorithm.ADAPTIVE,
        min_allocation: int = 1024,  # 1KB/s minimum
        reserve_percentage: float = 0.1,  # 10% reserve
        update_interval: float = 5.0,  # 5 seconds
    ) -> None:
        """
        Initialize enhanced bandwidth manager.
        
        Args:
            total_bandwidth: Total bandwidth limit in bytes per second (None for unlimited)
            algorithm: Allocation algorithm to use
            min_allocation: Minimum allocation per worker in bytes per second
            reserve_percentage: Percentage of bandwidth to keep in reserve
            update_interval: Interval between allocation updates in seconds
        """
        self.total_bandwidth = total_bandwidth
        self.algorithm = algorithm
        self.min_allocation = min_allocation
        self.reserve_percentage = reserve_percentage
        self.update_interval = update_interval
        
        # Worker allocations
        self.allocations: dict[str, BandwidthAllocation] = {}
        self.worker_priorities: dict[str, WorkerPriority] = {}
        self.worker_weights: dict[str, float] = {}
        
        # Performance tracking
        self.performance_history: deque[dict[str, Any]] = deque(maxlen=100)
        self.last_update = time.time()
        
        # Statistics
        self.allocation_count = 0
        self.reallocation_count = 0
        
        logger.info(
            f"Initialized EnhancedBandwidthManager: algorithm={algorithm.value}, "
            f"total_bandwidth={total_bandwidth}, min_allocation={min_allocation}"
        )
    
    def set_worker_priority(self, worker_id: str, priority: WorkerPriority) -> None:
        """Set priority for a worker."""
        self.worker_priorities[worker_id] = priority
        logger.debug(f"Set worker {worker_id} priority to {priority.value}")
        
    def set_worker_weight(self, worker_id: str, weight: float) -> None:
        """Set weight for a worker."""
        self.worker_weights[worker_id] = max(0.1, weight)  # Minimum weight of 0.1
        logger.debug(f"Set worker {worker_id} weight to {weight}")
    
    def add_worker(
        self,
        worker_id: str,
        priority: WorkerPriority = WorkerPriority.NORMAL,
        weight: float = 1.0,
    ) -> None:
        """Add a new worker to bandwidth management."""
        self.worker_priorities[worker_id] = priority
        self.worker_weights[worker_id] = weight
        
        # Initial allocation
        initial_allocation = self._calculate_initial_allocation()
        self.allocations[worker_id] = BandwidthAllocation(
            worker_id, initial_allocation, priority, weight
        )
        
        logger.debug(f"Added worker {worker_id} with initial allocation {initial_allocation} B/s")
        
    def remove_worker(self, worker_id: str) -> None:
        """Remove a worker from bandwidth management."""
        self.allocations.pop(worker_id, None)
        self.worker_priorities.pop(worker_id, None)
        self.worker_weights.pop(worker_id, None)
        
        logger.debug(f"Removed worker {worker_id}")
        
    def _calculate_initial_allocation(self) -> int:
        """Calculate initial allocation for a new worker."""
        if not self.total_bandwidth:
            return 0  # Unlimited
            
        active_workers = len(self.allocations) + 1  # +1 for the new worker
        available_bandwidth = self._get_available_bandwidth()
        
        return max(self.min_allocation, available_bandwidth // active_workers)
    
    def _get_available_bandwidth(self) -> int:
        """Get available bandwidth after reserves."""
        if not self.total_bandwidth:
            return 0  # Unlimited
            
        reserve = int(self.total_bandwidth * self.reserve_percentage)
        return max(0, self.total_bandwidth - reserve)
    
    def update_worker_usage(self, worker_id: str, current_speed: float) -> None:
        """Update current usage for a worker."""
        if worker_id in self.allocations:
            self.allocations[worker_id].update_usage(int(current_speed))
    
    def allocate_bandwidth(self, worker_speeds: dict[str, float]) -> dict[str, int]:
        """
        Allocate bandwidth among workers using the configured algorithm.
        
        Args:
            worker_speeds: Current speeds for each worker
            
        Returns:
            Dictionary mapping worker_id to allocated bandwidth
        """
        # Update usage statistics
        for worker_id, speed in worker_speeds.items():
            self.update_worker_usage(worker_id, speed)
        
        # Check if it's time to reallocate
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            # Return current allocations
            return {wid: alloc.allocated_bandwidth for wid, alloc in self.allocations.items()}
        
        # Perform reallocation
        new_allocations = self._perform_allocation(worker_speeds)
        
        # Update allocations
        for worker_id, bandwidth in new_allocations.items():
            if worker_id in self.allocations:
                old_allocation = self.allocations[worker_id].allocated_bandwidth
                self.allocations[worker_id].allocated_bandwidth = bandwidth
                self.allocations[worker_id].last_updated = current_time
                
                if old_allocation != bandwidth:
                    self.reallocation_count += 1
        
        self.last_update = current_time
        self.allocation_count += 1
        
        # Record performance metrics
        self._record_performance_metrics(worker_speeds, new_allocations)
        
        logger.debug(f"Allocated bandwidth using {self.algorithm.value}: {new_allocations}")
        return new_allocations
    
    def _perform_allocation(self, worker_speeds: dict[str, float]) -> dict[str, int]:
        """Perform bandwidth allocation using the configured algorithm."""
        if not worker_speeds:
            return {}
            
        if self.algorithm == AllocationAlgorithm.EQUAL:
            return self._equal_allocation(worker_speeds)
        elif self.algorithm == AllocationAlgorithm.PROPORTIONAL:
            return self._proportional_allocation(worker_speeds)
        elif self.algorithm == AllocationAlgorithm.WEIGHTED_FAIR:
            return self._weighted_fair_allocation(worker_speeds)
        elif self.algorithm == AllocationAlgorithm.PRIORITY_BASED:
            return self._priority_based_allocation(worker_speeds)
        elif self.algorithm == AllocationAlgorithm.ADAPTIVE:
            return self._adaptive_allocation(worker_speeds)
        else:
            return self._equal_allocation(worker_speeds)
    
    def _equal_allocation(self, worker_speeds: dict[str, float]) -> dict[str, int]:
        """Equal allocation among all workers."""
        if not self.total_bandwidth:
            return {worker_id: 0 for worker_id in worker_speeds}
            
        available = self._get_available_bandwidth()
        per_worker = max(self.min_allocation, available // len(worker_speeds))
        
        return {worker_id: per_worker for worker_id in worker_speeds}
    
    def _proportional_allocation(self, worker_speeds: dict[str, float]) -> dict[str, int]:
        """Proportional allocation based on current speeds."""
        if not self.total_bandwidth:
            return {worker_id: 0 for worker_id in worker_speeds}
            
        available = self._get_available_bandwidth()
        total_speed = sum(worker_speeds.values())
        
        if total_speed == 0:
            return self._equal_allocation(worker_speeds)
        
        allocations = {}
        remaining = available
        
        for worker_id, speed in worker_speeds.items():
            proportion = speed / total_speed
            allocation = max(self.min_allocation, int(available * proportion))
            allocations[worker_id] = allocation
            remaining -= allocation
        
        # Distribute any remaining bandwidth
        if remaining > 0:
            per_worker_bonus = remaining // len(worker_speeds)
            for worker_id in allocations:
                allocations[worker_id] += per_worker_bonus
        
        return allocations
    
    def _weighted_fair_allocation(self, worker_speeds: dict[str, float]) -> dict[str, int]:
        """Weighted fair queuing allocation."""
        if not self.total_bandwidth:
            return {worker_id: 0 for worker_id in worker_speeds}
            
        available = self._get_available_bandwidth()
        
        # Calculate total weight
        total_weight = sum(
            self.worker_weights.get(worker_id, 1.0) for worker_id in worker_speeds
        )
        
        if total_weight == 0:
            return self._equal_allocation(worker_speeds)
        
        allocations = {}
        for worker_id in worker_speeds:
            weight = self.worker_weights.get(worker_id, 1.0)
            allocation = max(self.min_allocation, int(available * weight / total_weight))
            allocations[worker_id] = allocation
        
        return allocations
    
    def _priority_based_allocation(self, worker_speeds: dict[str, float]) -> dict[str, int]:
        """Priority-based allocation."""
        if not self.total_bandwidth:
            return {worker_id: 0 for worker_id in worker_speeds}
            
        available = self._get_available_bandwidth()
        
        # Group workers by priority
        priority_groups = defaultdict(list)
        for worker_id in worker_speeds:
            priority = self.worker_priorities.get(worker_id, WorkerPriority.NORMAL)
            priority_groups[priority].append(worker_id)
        
        allocations = {}
        remaining = available
        
        # Allocate in priority order (highest first)
        for priority in sorted(priority_groups.keys(), key=lambda p: p.value, reverse=True):
            workers = priority_groups[priority]
            if not workers or remaining <= 0:
                continue
                
            # Allocate proportionally within priority group
            group_allocation = remaining // 2 if priority != WorkerPriority.LOW else remaining
            per_worker = max(self.min_allocation, group_allocation // len(workers))
            
            for worker_id in workers:
                allocation = min(per_worker, remaining)
                allocations[worker_id] = allocation
                remaining -= allocation
        
        # Ensure all workers get minimum allocation
        for worker_id in worker_speeds:
            if worker_id not in allocations:
                allocations[worker_id] = self.min_allocation
        
        return allocations
    
    def _adaptive_allocation(self, worker_speeds: dict[str, float]) -> dict[str, int]:
        """Adaptive allocation based on performance and utilization."""
        if not self.total_bandwidth:
            return {worker_id: 0 for worker_id in worker_speeds}
            
        available = self._get_available_bandwidth()
        
        # Calculate efficiency scores for each worker
        efficiency_scores = {}
        for worker_id in worker_speeds:
            if worker_id in self.allocations:
                allocation = self.allocations[worker_id]
                utilization = allocation.get_utilization_rate()
                avg_usage = allocation.get_average_usage()
                
                # Efficiency = utilization * consistency
                consistency = 1.0 - abs(allocation.actual_usage - avg_usage) / max(1, avg_usage)
                efficiency = utilization * max(0.1, consistency)
                efficiency_scores[worker_id] = efficiency
            else:
                efficiency_scores[worker_id] = 0.5  # Neutral for new workers
        
        # Allocate based on efficiency and current performance
        total_score = sum(efficiency_scores.values())
        if total_score == 0:
            return self._equal_allocation(worker_speeds)
        
        allocations = {}
        for worker_id, score in efficiency_scores.items():
            base_allocation = int(available * score / total_score)
            
            # Boost allocation for high-performing workers
            current_speed = worker_speeds.get(worker_id, 0)
            if current_speed > 0:
                speed_factor = min(2.0, math.log10(current_speed / 1024 + 1))  # Log scale
                base_allocation = int(base_allocation * speed_factor)
            
            allocations[worker_id] = max(self.min_allocation, base_allocation)
        
        # Normalize to fit within available bandwidth
        total_allocated = sum(allocations.values())
        if total_allocated > available:
            scale_factor = available / total_allocated
            for worker_id in allocations:
                allocations[worker_id] = max(
                    self.min_allocation, int(allocations[worker_id] * scale_factor)
                )
        
        return allocations
    
    def _record_performance_metrics(
        self, worker_speeds: dict[str, float], allocations: dict[str, int]
    ) -> None:
        """Record performance metrics for analysis."""
        total_speed = sum(worker_speeds.values())
        total_allocated = sum(allocations.values())
        
        metrics = {
            "timestamp": time.time(),
            "algorithm": self.algorithm.value,
            "worker_count": len(worker_speeds),
            "total_speed": total_speed,
            "total_allocated": total_allocated,
            "utilization": total_speed / max(1, total_allocated) if total_allocated > 0 else 0,
            "efficiency_scores": {},
        }
        
        # Calculate per-worker efficiency
        for worker_id in worker_speeds:
            if worker_id in self.allocations:
                allocation = self.allocations[worker_id]
                metrics["efficiency_scores"][worker_id] = allocation.get_utilization_rate()
        
        self.performance_history.append(metrics)
    
    def get_allocation_stats(self) -> dict[str, Any]:
        """Get comprehensive allocation statistics."""
        current_time = time.time()
        
        stats = {
            "algorithm": self.algorithm.value,
            "total_bandwidth": self.total_bandwidth,
            "available_bandwidth": self._get_available_bandwidth(),
            "worker_count": len(self.allocations),
            "allocation_count": self.allocation_count,
            "reallocation_count": self.reallocation_count,
            "last_update": self.last_update,
            "time_since_update": current_time - self.last_update,
            "workers": {},
        }
        
        # Per-worker statistics
        total_allocated = 0
        total_usage = 0
        
        for worker_id, allocation in self.allocations.items():
            worker_stats = {
                "allocated_bandwidth": allocation.allocated_bandwidth,
                "actual_usage": allocation.actual_usage,
                "utilization_rate": allocation.get_utilization_rate(),
                "average_usage": allocation.get_average_usage(),
                "priority": self.worker_priorities.get(worker_id, WorkerPriority.NORMAL).name,
                "weight": self.worker_weights.get(worker_id, 1.0),
                "last_updated": allocation.last_updated,
            }
            stats["workers"][worker_id] = worker_stats
            
            total_allocated += allocation.allocated_bandwidth
            total_usage += allocation.actual_usage
        
        stats["total_allocated"] = total_allocated
        stats["total_usage"] = total_usage
        stats["overall_utilization"] = total_usage / max(1, total_allocated)
        
        return stats
    
    def get_performance_analysis(self) -> dict[str, Any]:
        """Get performance analysis over time."""
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        recent_metrics = list(self.performance_history)[-10:]  # Last 10 measurements
        
        # Calculate trends
        utilizations = [m["utilization"] for m in recent_metrics]
        speeds = [m["total_speed"] for m in recent_metrics]
        
        analysis = {
            "measurement_count": len(self.performance_history),
            "recent_measurements": len(recent_metrics),
            "average_utilization": sum(utilizations) / len(utilizations),
            "average_total_speed": sum(speeds) / len(speeds),
            "utilization_trend": "stable",
            "speed_trend": "stable",
            "algorithm_effectiveness": "good",
        }
        
        # Determine trends
        if len(utilizations) >= 3:
            if utilizations[-1] > utilizations[0] * 1.1:
                analysis["utilization_trend"] = "increasing"
            elif utilizations[-1] < utilizations[0] * 0.9:
                analysis["utilization_trend"] = "decreasing"
        
        if len(speeds) >= 3:
            if speeds[-1] > speeds[0] * 1.1:
                analysis["speed_trend"] = "increasing"
            elif speeds[-1] < speeds[0] * 0.9:
                analysis["speed_trend"] = "decreasing"
        
        # Algorithm effectiveness
        avg_util = analysis["average_utilization"]
        if avg_util > 0.8:
            analysis["algorithm_effectiveness"] = "excellent"
        elif avg_util > 0.6:
            analysis["algorithm_effectiveness"] = "good"
        elif avg_util > 0.4:
            analysis["algorithm_effectiveness"] = "fair"
        else:
            analysis["algorithm_effectiveness"] = "poor"
        
        return analysis
    
    def optimize_allocation(self) -> dict[str, Any]:
        """Suggest optimizations for current allocation."""
        stats = self.get_allocation_stats()
        suggestions = []
        
        # Check overall utilization
        overall_util = stats["overall_utilization"]
        if overall_util < 0.5:
            suggestions.append("Consider reducing total bandwidth limit or switching to proportional allocation")
        elif overall_util > 0.9:
            suggestions.append("Consider increasing total bandwidth limit or using priority-based allocation")
        
        # Check per-worker utilization
        underutilized_workers = []
        overutilized_workers = []
        
        for worker_id, worker_stats in stats["workers"].items():
            util_rate = worker_stats["utilization_rate"]
            if util_rate < 0.3:
                underutilized_workers.append(worker_id)
            elif util_rate > 0.95:
                overutilized_workers.append(worker_id)
        
        if underutilized_workers:
            suggestions.append(f"Workers {underutilized_workers} are underutilized - consider reducing their allocation")
        
        if overutilized_workers:
            suggestions.append(f"Workers {overutilized_workers} are overutilized - consider increasing their allocation")
        
        # Algorithm recommendations
        if self.algorithm == AllocationAlgorithm.EQUAL and len(set(
            w["utilization_rate"] for w in stats["workers"].values()
        )) > 1:
            suggestions.append("Consider switching to adaptive or proportional allocation for better efficiency")
        
        return {
            "current_algorithm": self.algorithm.value,
            "overall_utilization": overall_util,
            "suggestions": suggestions,
            "underutilized_workers": underutilized_workers,
            "overutilized_workers": overutilized_workers,
        }