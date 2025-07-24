from fastapi import FastAPI, APIRouter, HTTPException, Depends
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from collections import defaultdict
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="The Third Angle API", version="2.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Enums
class TaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class GoalType(str, Enum):
    TASK_BASED = "task_based"
    TIME_BASED = "time_based"
    OKR = "okr"

class NotificationType(str, Enum):
    TASK_ASSIGNED = "task_assigned"
    TASK_UPDATED = "task_updated"
    TASK_COMPLETED = "task_completed"
    DEADLINE_REMINDER = "deadline_reminder"
    MENTION = "mention"

# Enhanced Models
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str
    avatar_url: Optional[str] = None
    role: str = "team_member"
    joined_date: datetime = Field(default_factory=datetime.utcnow)
    productivity_score: float = 0.0
    total_tasks_completed: int = 0
    total_hours_logged: float = 0.0
    burnout_risk: str = "low"  # low, medium, high
    badges: List[str] = []
    preferences: Dict[str, Any] = {}

class UserCreate(BaseModel):
    name: str
    email: str
    avatar_url: Optional[str] = None

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.TODO
    priority: Priority = Priority.MEDIUM
    assigned_to: Optional[str] = None  # Made optional for unassigned tasks
    assigned_users: List[str] = []  # Support multiple assignees
    project_id: Optional[str] = None
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None
    created_date: datetime = Field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = None
    completed_date: Optional[datetime] = None
    tags: List[str] = []
    position: int = 0  # For drag-and-drop ordering
    source: Optional[str] = None  # github, notion, manual
    source_url: Optional[str] = None
    comments_count: int = 0
    watchers: List[str] = []

class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    priority: Priority = Priority.MEDIUM
    assigned_to: Optional[str] = None
    assigned_users: List[str] = []
    project_id: Optional[str] = None
    estimated_hours: Optional[float] = None
    due_date: Optional[datetime] = None
    tags: List[str] = []
    source: Optional[str] = "manual"
    source_url: Optional[str] = None

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[TaskStatus] = None
    priority: Optional[Priority] = None
    assigned_to: Optional[str] = None
    assigned_users: Optional[List[str]] = None
    actual_hours: Optional[float] = None
    due_date: Optional[datetime] = None
    tags: Optional[List[str]] = None
    position: Optional[int] = None

class TaskComment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    user_id: str
    content: str
    mentions: List[str] = []  # User IDs mentioned in comment
    created_date: datetime = Field(default_factory=datetime.utcnow)
    updated_date: Optional[datetime] = None

class TaskCommentCreate(BaseModel):
    task_id: str
    user_id: str
    content: str
    mentions: List[str] = []

class TimeEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    task_id: Optional[str] = None
    description: str
    hours: float
    date: datetime = Field(default_factory=datetime.utcnow)
    is_pomodoro: bool = False
    is_overtime: bool = False  # Flag for burnout detection

class TimeEntryCreate(BaseModel):
    user_id: str
    task_id: Optional[str] = None
    description: str
    hours: float
    is_pomodoro: bool = False

class Goal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str
    description: Optional[str] = None
    goal_type: GoalType
    target_value: float
    current_value: float = 0.0
    deadline: Optional[datetime] = None
    created_date: datetime = Field(default_factory=datetime.utcnow)
    completed: bool = False

class GoalCreate(BaseModel):
    user_id: str
    title: str
    description: Optional[str] = None
    goal_type: GoalType
    target_value: float
    deadline: Optional[datetime] = None

class DailyStandup(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    date: datetime = Field(default_factory=datetime.utcnow)
    what_i_did: str
    what_ill_do: str
    blockers: Optional[str] = None

class DailyStandupCreate(BaseModel):
    user_id: str
    what_i_did: str
    what_ill_do: str
    blockers: Optional[str] = None

class Notification(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str
    message: str
    type: NotificationType
    read: bool = False
    created_date: datetime = Field(default_factory=datetime.utcnow)
    related_task_id: Optional[str] = None
    related_user_id: Optional[str] = None

class NotificationCreate(BaseModel):
    user_id: str
    title: str
    message: str
    type: NotificationType
    related_task_id: Optional[str] = None
    related_user_id: Optional[str] = None

class WikiPage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    author_id: str
    created_date: datetime = Field(default_factory=datetime.utcnow)
    updated_date: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = []
    is_public: bool = True

class WikiPageCreate(BaseModel):
    title: str
    content: str
    author_id: str
    tags: List[str] = []
    is_public: bool = True

# Helper functions
async def create_notification(user_id: str, title: str, message: str, notification_type: NotificationType, task_id: str = None, related_user_id: str = None):
    """Create a new notification"""
    notification = Notification(
        user_id=user_id,
        title=title,
        message=message,
        type=notification_type,
        related_task_id=task_id,
        related_user_id=related_user_id
    )
    await db.notifications.insert_one(notification.dict())
    return notification

async def calculate_burnout_risk(user_id: str) -> str:
    """Calculate burnout risk based on working patterns"""
    # Get last 14 days of time entries
    two_weeks_ago = datetime.utcnow() - timedelta(days=14)
    time_entries = await db.time_entries.find({
        "user_id": user_id,
        "date": {"$gte": two_weeks_ago}
    }).to_list(1000)
    
    if not time_entries:
        return "low"
    
    # Calculate daily hours
    daily_hours = defaultdict(float)
    for entry in time_entries:
        day = entry["date"].date()
        daily_hours[day] += entry["hours"]
    
    # Calculate metrics
    avg_daily_hours = sum(daily_hours.values()) / len(daily_hours) if daily_hours else 0
    max_daily_hours = max(daily_hours.values()) if daily_hours else 0
    days_over_8_hours = sum(1 for hours in daily_hours.values() if hours > 8)
    
    # Burnout risk calculation
    if avg_daily_hours > 9 or max_daily_hours > 12 or days_over_8_hours > 7:
        return "high"
    elif avg_daily_hours > 7.5 or max_daily_hours > 10 or days_over_8_hours > 3:
        return "medium"
    else:
        return "low"

async def update_user_badges(user_id: str):
    """Update user badges based on achievements"""
    user = await db.users.find_one({"id": user_id})
    if not user:
        return
    
    badges = set(user.get("badges", []))
    
    # Check for task completion badges
    completed_tasks = await db.tasks.count_documents({
        "assigned_to": user_id,
        "status": TaskStatus.DONE
    })
    
    if completed_tasks >= 10:
        badges.add("task_master_10")
    if completed_tasks >= 50:
        badges.add("task_master_50")
    if completed_tasks >= 100:
        badges.add("task_master_100")
    
    # Check for consistency badges
    week_ago = datetime.utcnow() - timedelta(days=7)
    daily_activity = await db.time_entries.aggregate([
        {
            "$match": {
                "user_id": user_id,
                "date": {"$gte": week_ago}
            }
        },
        {
            "$group": {
                "_id": {
                    "$dateToString": {
                        "format": "%Y-%m-%d",
                        "date": "$date"
                    }
                },
                "count": {"$sum": 1}
            }
        }
    ]).to_list(1000)
    
    if len(daily_activity) >= 7:
        badges.add("consistent_7_days")
    
    # Update user badges
    await db.users.update_one(
        {"id": user_id},
        {"$set": {"badges": list(badges)}}
    )

# User routes
@api_router.post("/users", response_model=User)
async def create_user(user_data: UserCreate):
    # Check if email already exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(**user_data.dict())
    await db.users.insert_one(user.dict())
    return user

@api_router.get("/users", response_model=List[User])
async def get_users():
    users = await db.users.find().to_list(1000)
    return [User(**user) for user in users]

@api_router.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str):
    user = await db.users.find_one({"id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return User(**user)

# Enhanced Task routes
@api_router.post("/tasks", response_model=Task)
async def create_task(task_data: TaskCreate):
    # Verify assigned users exist
    if task_data.assigned_to:
        user = await db.users.find_one({"id": task_data.assigned_to})
        if not user:
            raise HTTPException(status_code=404, detail="Assigned user not found")
    
    for user_id in task_data.assigned_users:
        user = await db.users.find_one({"id": user_id})
        if not user:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    # Set position for new task
    last_task = await db.tasks.find_one({}, sort=[("position", -1)])
    position = (last_task["position"] + 1) if last_task else 0
    
    task_dict = task_data.dict()
    task_dict["position"] = position
    task = Task(**task_dict)
    await db.tasks.insert_one(task.dict())
    
    # Create notifications for assigned users
    all_assigned = []
    if task_data.assigned_to:
        all_assigned.append(task_data.assigned_to)
    all_assigned.extend(task_data.assigned_users)
    
    for user_id in set(all_assigned):  # Remove duplicates
        await create_notification(
            user_id=user_id,
            title="New Task Assigned",
            message=f"You have been assigned to task: {task.title}",
            notification_type=NotificationType.TASK_ASSIGNED,
            task_id=task.id
        )
    
    return task

@api_router.get("/tasks", response_model=List[Task])
async def get_tasks(
    user_id: Optional[str] = None, 
    status: Optional[TaskStatus] = None,
    project_id: Optional[str] = None,
    unassigned: Optional[bool] = None
):
    query = {}
    if user_id:
        query["$or"] = [
            {"assigned_to": user_id},
            {"assigned_users": {"$in": [user_id]}}
        ]
    if status:
        query["status"] = status
    if project_id:
        query["project_id"] = project_id
    if unassigned:
        query["assigned_to"] = None
        query["assigned_users"] = {"$size": 0}
    
    tasks = await db.tasks.find(query).sort("position", 1).to_list(1000)
    return [Task(**task) for task in tasks]

@api_router.get("/tasks/kanban")
async def get_kanban_tasks():
    """Get tasks organized by status for Kanban board"""
    tasks = await db.tasks.find().sort("position", 1).to_list(1000)
    
    kanban_data = {
        "todo": [],
        "in_progress": [],
        "done": [],
        "blocked": []
    }
    
    for task in tasks:
        task_obj = Task(**task)
        kanban_data[task_obj.status].append(task_obj.dict())
    
    return kanban_data

@api_router.put("/tasks/{task_id}", response_model=Task)
async def update_task(task_id: str, task_update: TaskUpdate):
    task = await db.tasks.find_one({"id": task_id})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    update_data = {k: v for k, v in task_update.dict().items() if v is not None}
    
    # If status is being updated to done, record completion date
    if update_data.get("status") == TaskStatus.DONE and task["status"] != TaskStatus.DONE:
        update_data["completed_date"] = datetime.utcnow()
        
        # Update user's completed tasks count and badges
        if task.get("assigned_to"):
            await db.users.update_one(
                {"id": task["assigned_to"]}, 
                {"$inc": {"total_tasks_completed": 1}}
            )
            await update_user_badges(task["assigned_to"])
            
            # Create completion notification
            await create_notification(
                user_id=task["assigned_to"],
                title="Task Completed!",
                message=f"Great job completing: {task['title']}",
                notification_type=NotificationType.TASK_COMPLETED,
                task_id=task_id
            )
    
    # Handle assignment changes
    if "assigned_to" in update_data or "assigned_users" in update_data:
        new_assigned = update_data.get("assigned_to")
        new_assigned_users = update_data.get("assigned_users", [])
        
        # Get current assignments
        current_assigned = task.get("assigned_to")
        current_assigned_users = task.get("assigned_users", [])
        
        # Find newly assigned users
        all_new = []
        if new_assigned and new_assigned != current_assigned:
            all_new.append(new_assigned)
        for user_id in new_assigned_users:
            if user_id not in current_assigned_users:
                all_new.append(user_id)
        
        # Create notifications for newly assigned users
        for user_id in set(all_new):
            await create_notification(
                user_id=user_id,
                title="Task Assignment Updated",
                message=f"You have been assigned to task: {task['title']}",
                notification_type=NotificationType.TASK_ASSIGNED,
                task_id=task_id
            )
    
    await db.tasks.update_one({"id": task_id}, {"$set": update_data})
    
    updated_task = await db.tasks.find_one({"id": task_id})
    return Task(**updated_task)

@api_router.put("/tasks/bulk-update-positions")
async def bulk_update_task_positions(updates: List[Dict[str, Any]]):
    """Bulk update task positions for drag-and-drop"""
    for update in updates:
        await db.tasks.update_one(
            {"id": update["id"]},
            {"$set": {"position": update["position"], "status": update.get("status", "todo")}}
        )
    
    return {"message": "Task positions updated successfully"}

@api_router.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    result = await db.tasks.delete_one({"id": task_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"message": "Task deleted successfully"}

# Task Comments routes
@api_router.post("/tasks/{task_id}/comments", response_model=TaskComment)
async def create_task_comment(task_id: str, comment_data: TaskCommentCreate):
    # Verify task exists
    task = await db.tasks.find_one({"id": task_id})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    comment = TaskComment(**comment_data.dict())
    await db.task_comments.insert_one(comment.dict())
    
    # Update task comment count
    await db.tasks.update_one(
        {"id": task_id},
        {"$inc": {"comments_count": 1}}
    )
    
    # Create notifications for mentioned users
    for mentioned_user_id in comment_data.mentions:
        user = await db.users.find_one({"id": mentioned_user_id})
        if user:
            await create_notification(
                user_id=mentioned_user_id,
                title="You were mentioned",
                message=f"You were mentioned in a comment on task: {task['title']}",
                notification_type=NotificationType.MENTION,
                task_id=task_id,
                related_user_id=comment_data.user_id
            )
    
    return comment

@api_router.get("/tasks/{task_id}/comments", response_model=List[TaskComment])
async def get_task_comments(task_id: str):
    comments = await db.task_comments.find({"task_id": task_id}).sort("created_date", 1).to_list(1000)
    return [TaskComment(**comment) for comment in comments]

# Time tracking routes
@api_router.post("/time-entries", response_model=TimeEntry)
async def create_time_entry(time_data: TimeEntryCreate):
    # Check for overtime (burnout risk)
    is_overtime = time_data.hours > 8
    
    time_entry_dict = time_data.dict()
    time_entry_dict["is_overtime"] = is_overtime
    time_entry = TimeEntry(**time_entry_dict)
    await db.time_entries.insert_one(time_entry.dict())
    
    # Update user's total hours
    await db.users.update_one(
        {"id": time_data.user_id},
        {"$inc": {"total_hours_logged": time_data.hours}}
    )
    
    # Update task's actual hours if task_id provided
    if time_data.task_id:
        await db.tasks.update_one(
            {"id": time_data.task_id},
            {"$inc": {"actual_hours": time_data.hours}}
        )
    
    # Update burnout risk
    burnout_risk = await calculate_burnout_risk(time_data.user_id)
    await db.users.update_one(
        {"id": time_data.user_id},
        {"$set": {"burnout_risk": burnout_risk}}
    )
    
    return time_entry

@api_router.get("/time-entries", response_model=List[TimeEntry])
async def get_time_entries(user_id: Optional[str] = None, task_id: Optional[str] = None):
    query = {}
    if user_id:
        query["user_id"] = user_id
    if task_id:
        query["task_id"] = task_id
    
    entries = await db.time_entries.find(query).sort("date", -1).to_list(1000)
    return [TimeEntry(**entry) for entry in entries]

# Goals routes
@api_router.post("/goals", response_model=Goal)
async def create_goal(goal_data: GoalCreate):
    goal = Goal(**goal_data.dict())
    await db.goals.insert_one(goal.dict())
    return goal

@api_router.get("/goals", response_model=List[Goal])
async def get_goals(user_id: Optional[str] = None):
    query = {}
    if user_id:
        query["user_id"] = user_id
    
    goals = await db.goals.find(query).to_list(1000)
    return [Goal(**goal) for goal in goals]

# Standup routes
@api_router.post("/standups", response_model=DailyStandup)
async def create_standup(standup_data: DailyStandupCreate):
    # Check if user already has standup for today
    today = datetime.utcnow().date()
    existing = await db.standups.find_one({
        "user_id": standup_data.user_id,
        "date": {"$gte": datetime.combine(today, datetime.min.time())}
    })
    
    if existing:
        raise HTTPException(status_code=400, detail="Standup already exists for today")
    
    standup = DailyStandup(**standup_data.dict())
    await db.standups.insert_one(standup.dict())
    return standup

@api_router.get("/standups", response_model=List[DailyStandup])
async def get_standups(user_id: Optional[str] = None, date: Optional[datetime] = None):
    query = {}
    if user_id:
        query["user_id"] = user_id
    if date:
        start_date = datetime.combine(date.date(), datetime.min.time())
        end_date = start_date + timedelta(days=1)
        query["date"] = {"$gte": start_date, "$lt": end_date}
    
    standups = await db.standups.find(query).sort("date", -1).to_list(1000)
    return [DailyStandup(**standup) for standup in standups]

# Notifications routes
@api_router.get("/notifications/{user_id}", response_model=List[Notification])
async def get_user_notifications(user_id: str, unread_only: bool = False):
    query = {"user_id": user_id}
    if unread_only:
        query["read"] = False
    
    notifications = await db.notifications.find(query).sort("created_date", -1).to_list(100)
    return [Notification(**notification) for notification in notifications]

@api_router.put("/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str):
    result = await db.notifications.update_one(
        {"id": notification_id},
        {"$set": {"read": True}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"message": "Notification marked as read"}

# Wiki routes
@api_router.post("/wiki", response_model=WikiPage)
async def create_wiki_page(page_data: WikiPageCreate):
    page = WikiPage(**page_data.dict())
    await db.wiki_pages.insert_one(page.dict())
    return page

@api_router.get("/wiki", response_model=List[WikiPage])
async def get_wiki_pages():
    pages = await db.wiki_pages.find({"is_public": True}).sort("updated_date", -1).to_list(1000)
    return [WikiPage(**page) for page in pages]

@api_router.get("/wiki/{page_id}", response_model=WikiPage)
async def get_wiki_page(page_id: str):
    page = await db.wiki_pages.find_one({"id": page_id})
    if not page:
        raise HTTPException(status_code=404, detail="Wiki page not found")
    return WikiPage(**page)

# Enhanced Analytics routes
@api_router.get("/analytics/team-overview")
async def get_team_overview():
    # Get all users
    users = await db.users.find().to_list(1000)
    
    # Get tasks stats
    total_tasks = await db.tasks.count_documents({})
    completed_tasks = await db.tasks.count_documents({"status": TaskStatus.DONE})
    in_progress_tasks = await db.tasks.count_documents({"status": TaskStatus.IN_PROGRESS})
    blocked_tasks = await db.tasks.count_documents({"status": TaskStatus.BLOCKED})
    unassigned_tasks = await db.tasks.count_documents({"assigned_to": None})
    
    # Get today's productivity
    today = datetime.utcnow().date()
    today_tasks = await db.tasks.count_documents({
        "completed_date": {"$gte": datetime.combine(today, datetime.min.time())}
    })
    
    # Calculate team productivity score
    productivity_score = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    # Get burnout risks
    high_burnout_users = await db.users.count_documents({"burnout_risk": "high"})
    medium_burnout_users = await db.users.count_documents({"burnout_risk": "medium"})
    
    return {
        "team_size": len(users),
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "in_progress_tasks": in_progress_tasks,
        "blocked_tasks": blocked_tasks,
        "unassigned_tasks": unassigned_tasks,
        "tasks_completed_today": today_tasks,
        "team_productivity_score": round(productivity_score, 1),
        "completion_rate": round((completed_tasks / total_tasks * 100) if total_tasks > 0 else 0, 1),
        "high_burnout_users": high_burnout_users,
        "medium_burnout_users": medium_burnout_users
    }

@api_router.get("/analytics/individual-performance")
async def get_individual_performance():
    users = await db.users.find().to_list(1000)
    performance_data = []
    
    for user in users:
        user_tasks = await db.tasks.count_documents({
            "$or": [
                {"assigned_to": user["id"]},
                {"assigned_users": {"$in": [user["id"]]}}
            ]
        })
        completed_tasks = await db.tasks.count_documents({
            "$or": [
                {"assigned_to": user["id"]},
                {"assigned_users": {"$in": [user["id"]]}}
            ],
            "status": TaskStatus.DONE
        })
        
        # Get time entries for the last 7 days
        week_ago = datetime.utcnow() - timedelta(days=7)
        time_entries = await db.time_entries.find({
            "user_id": user["id"],
            "date": {"$gte": week_ago}
        }).to_list(1000)
        
        hours_this_week = sum(entry["hours"] for entry in time_entries)
        
        # Calculate productivity score
        completion_rate = (completed_tasks / user_tasks * 100) if user_tasks > 0 else 0
        productivity_score = (completion_rate + min(hours_this_week * 2, 100)) / 2
        
        performance_data.append({
            "user_id": user["id"],
            "name": user["name"],
            "avatar_url": user.get("avatar_url"),
            "total_tasks": user_tasks,
            "completed_tasks": completed_tasks,
            "completion_rate": round(completion_rate, 1),
            "hours_this_week": round(hours_this_week, 1),
            "productivity_score": round(productivity_score, 1),
            "burnout_risk": user.get("burnout_risk", "low"),
            "badges": user.get("badges", [])
        })
    
    # Sort by productivity score
    performance_data.sort(key=lambda x: x["productivity_score"], reverse=True)
    return performance_data

@api_router.get("/analytics/productivity-trends")
async def get_productivity_trends():
    # Get last 30 days of data
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    
    # Get daily task completions
    pipeline = [
        {
            "$match": {
                "completed_date": {"$gte": thirty_days_ago},
                "status": TaskStatus.DONE
            }
        },
        {
            "$group": {
                "_id": {
                    "$dateToString": {
                        "format": "%Y-%m-%d",
                        "date": "$completed_date"
                    }
                },
                "count": {"$sum": 1}
            }
        },
        {"$sort": {"_id": 1}}
    ]
    
    task_trends = await db.tasks.aggregate(pipeline).to_list(30)
    
    # Get daily time entries
    time_pipeline = [
        {
            "$match": {
                "date": {"$gte": thirty_days_ago}
            }
        },
        {
            "$group": {
                "_id": {
                    "$dateToString": {
                        "format": "%Y-%m-%d",
                        "date": "$date"
                    }
                },
                "total_hours": {"$sum": "$hours"},
                "overtime_hours": {
                    "$sum": {
                        "$cond": ["$is_overtime", "$hours", 0]
                    }
                }
            }
        },
        {"$sort": {"_id": 1}}
    ]
    
    time_trends = await db.time_entries.aggregate(time_pipeline).to_list(30)
    
    return {
        "task_completion_trends": task_trends,
        "time_logging_trends": time_trends
    }

@api_router.get("/analytics/team-leaderboard")
async def get_team_leaderboard():
    users = await db.users.find().to_list(1000)
    leaderboard = []
    
    for user in users:
        # Get tasks completed this month
        current_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        monthly_tasks = await db.tasks.count_documents({
            "$or": [
                {"assigned_to": user["id"]},
                {"assigned_users": {"$in": [user["id"]]}}
            ],
            "status": TaskStatus.DONE,
            "completed_date": {"$gte": current_month}
        })
        
        # Get hours logged this month
        time_entries = await db.time_entries.find({
            "user_id": user["id"],
            "date": {"$gte": current_month}
        }).to_list(1000)
        
        monthly_hours = sum(entry["hours"] for entry in time_entries)
        
        # Calculate points (tasks completed * 10 + hours * 2)
        points = monthly_tasks * 10 + monthly_hours * 2
        
        leaderboard.append({
            "user_id": user["id"],
            "name": user["name"],
            "avatar_url": user.get("avatar_url"),
            "tasks_completed": monthly_tasks,
            "hours_logged": round(monthly_hours, 1),
            "points": round(points, 1),
            "rank": 0,  # Will be set after sorting
            "badges": user.get("badges", []),
            "burnout_risk": user.get("burnout_risk", "low")
        })
    
    # Sort by points and assign ranks
    leaderboard.sort(key=lambda x: x["points"], reverse=True)
    for i, user in enumerate(leaderboard):
        user["rank"] = i + 1
    
    return leaderboard

@api_router.get("/analytics/burnout-analysis")
async def get_burnout_analysis():
    """Get burnout analysis for the team"""
    users = await db.users.find().to_list(1000)
    burnout_data = []
    
    for user in users:
        # Get recent activity
        week_ago = datetime.utcnow() - timedelta(days=7)
        time_entries = await db.time_entries.find({
            "user_id": user["id"],
            "date": {"$gte": week_ago}
        }).to_list(1000)
        
        # Calculate metrics
        total_hours = sum(entry["hours"] for entry in time_entries)
        overtime_hours = sum(entry["hours"] for entry in time_entries if entry.get("is_overtime", False))
        avg_daily_hours = total_hours / 7 if total_hours > 0 else 0
        
        burnout_data.append({
            "user_id": user["id"],
            "name": user["name"],
            "avatar_url": user.get("avatar_url"),
            "burnout_risk": user.get("burnout_risk", "low"),
            "total_hours_week": round(total_hours, 1),
            "overtime_hours_week": round(overtime_hours, 1),
            "avg_daily_hours": round(avg_daily_hours, 1)
        })
    
    return burnout_data

# Initialize with enhanced sample data
@api_router.post("/init-sample-data")
async def init_sample_data():
    # Clear existing data
    await db.users.delete_many({})
    await db.tasks.delete_many({})
    await db.time_entries.delete_many({})
    await db.goals.delete_many({})
    await db.standups.delete_many({})
    await db.notifications.delete_many({})
    await db.task_comments.delete_many({})
    await db.wiki_pages.delete_many({})
    
    # Create sample users with enhanced data
    sample_users = [
        {"name": "Alex Johnson", "email": "alex@thirdangle.com", "avatar_url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150", "role": "team_lead"},
        {"name": "Sarah Chen", "email": "sarah@thirdangle.com", "avatar_url": "https://images.unsplash.com/photo-1494790108755-2616b612b786?w=150", "role": "developer"},
        {"name": "Mike Rodriguez", "email": "mike@thirdangle.com", "avatar_url": "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150", "role": "designer"},
        {"name": "Emma Wilson", "email": "emma@thirdangle.com", "avatar_url": "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=150", "role": "product_manager"},
        {"name": "David Kim", "email": "david@thirdangle.com", "avatar_url": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=150", "role": "developer"},
        {"name": "Lisa Zhang", "email": "lisa@thirdangle.com", "avatar_url": "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=150", "role": "qa_engineer"}
    ]
    
    user_ids = []
    for user_data in sample_users:
        user = User(**user_data)
        await db.users.insert_one(user.dict())
        user_ids.append(user.id)
    
    # Create enhanced sample tasks
    task_templates = [
        {"title": "Design Landing Page", "description": "Create wireframes and mockups for the new landing page", "priority": Priority.HIGH, "source": "manual"},
        {"title": "API Development", "description": "Build REST endpoints for user management", "priority": Priority.HIGH, "source": "manual"},
        {"title": "User Authentication", "description": "Implement OAuth2 login system", "priority": Priority.MEDIUM, "source": "github", "source_url": "https://github.com/example/repo/issues/123"},
        {"title": "Database Migration", "description": "Update schema to support new features", "priority": Priority.LOW, "source": "manual"},
        {"title": "Testing Suite", "description": "Write comprehensive unit tests", "priority": Priority.MEDIUM, "source": "manual"},
        {"title": "UI Components", "description": "Build reusable React components", "priority": Priority.HIGH, "source": "notion", "source_url": "https://notion.so/example-page"},
        {"title": "Performance Optimization", "description": "Improve page load times", "priority": Priority.LOW, "source": "manual"},
        {"title": "Documentation", "description": "Write API documentation", "priority": Priority.MEDIUM, "source": "manual"},
        {"title": "Code Review", "description": "Review pending pull requests", "priority": Priority.HIGH, "source": "github", "source_url": "https://github.com/example/repo/pulls"},
        {"title": "Bug Fixes", "description": "Fix reported critical issues", "priority": Priority.MEDIUM, "source": "manual"},
        {"title": "Mobile Responsiveness", "description": "Ensure app works on mobile devices", "priority": Priority.HIGH, "source": "manual"},
        {"title": "Analytics Integration", "description": "Add Google Analytics tracking", "priority": Priority.LOW, "source": "manual"},
    ]
    
    # Create tasks with various statuses and assignments
    task_ids = []
    for i, template in enumerate(task_templates):
        for j in range(2):  # Create 2 tasks per template
            created_date = datetime.utcnow() - timedelta(days=30-i*2-j)
            assigned_user = user_ids[(i + j) % len(user_ids)]
            
            task_data = {
                **template,
                "assigned_to": assigned_user,
                "created_date": created_date,
                "estimated_hours": 4.0 + (i % 5),
                "position": len(task_ids),
                "tags": ["development", "urgent"] if template["priority"] == Priority.HIGH else ["development"]
            }
            
            # Assign status and completion based on pattern
            if (i + j) % 4 == 0:  # 25% completed
                task_data["status"] = TaskStatus.DONE
                task_data["completed_date"] = created_date + timedelta(days=1+i%5)
                task_data["actual_hours"] = task_data["estimated_hours"] + (i % 3 - 1)
            elif (i + j) % 4 == 1:  # 25% in progress
                task_data["status"] = TaskStatus.IN_PROGRESS
                task_data["actual_hours"] = task_data["estimated_hours"] / 2
            elif (i + j) % 4 == 2:  # 25% blocked
                task_data["status"] = TaskStatus.BLOCKED
            # else: 25% TODO (default)
            
            # Some tasks have multiple assignees
            if i % 3 == 0:
                task_data["assigned_users"] = [user_ids[(i + j + 1) % len(user_ids)]]
            
            task = Task(**task_data)
            await db.tasks.insert_one(task.dict())
            task_ids.append(task.id)
    
    # Create sample time entries with realistic patterns
    for user_id in user_ids:
        user_index = user_ids.index(user_id)
        for day in range(30):
            date = datetime.utcnow() - timedelta(days=day)
            if date.weekday() < 5:  # Weekdays only
                # Vary hours by user to create burnout scenarios
                if user_index == 0:  # Alex - high hours (burnout risk)
                    morning_hours = 4.5 + (day % 2) * 0.5
                    afternoon_hours = 5.0 + (day % 3) * 0.5
                elif user_index == 1:  # Sarah - moderate hours
                    morning_hours = 3.5 + (day % 3) * 0.5
                    afternoon_hours = 4.0 + (day % 2) * 0.5
                else:  # Others - normal hours
                    morning_hours = 3.0 + (day % 3) * 0.5
                    afternoon_hours = 3.5 + (day % 2) * 0.5
                
                # Morning session
                morning_entry = TimeEntry(
                    user_id=user_id,
                    task_id=task_ids[(day + user_index) % len(task_ids)] if task_ids else None,
                    description=f"Morning development work",
                    hours=morning_hours,
                    date=date.replace(hour=9),
                    is_pomodoro=True,
                    is_overtime=morning_hours > 4
                )
                await db.time_entries.insert_one(morning_entry.dict())
                
                # Afternoon session
                afternoon_entry = TimeEntry(
                    user_id=user_id,
                    task_id=task_ids[(day + user_index + 1) % len(task_ids)] if task_ids else None,
                    description=f"Afternoon project work",
                    hours=afternoon_hours,
                    date=date.replace(hour=14),
                    is_overtime=afternoon_hours > 4
                )
                await db.time_entries.insert_one(afternoon_entry.dict())
    
    # Create sample comments
    for i, task_id in enumerate(task_ids[:5]):  # Add comments to first 5 tasks
        comment = TaskComment(
            task_id=task_id,
            user_id=user_ids[i % len(user_ids)],
            content=f"This task is progressing well. @{user_ids[(i+1) % len(user_ids)]} please review when ready.",
            mentions=[user_ids[(i+1) % len(user_ids)]]
        )
        await db.task_comments.insert_one(comment.dict())
    
    # Create sample wiki pages
    wiki_pages = [
        {
            "title": "Team Onboarding Guide",
            "content": "# Welcome to The Third Angle Team\n\nThis guide will help you get started with our productivity workflow...",
            "author_id": user_ids[0],
            "tags": ["onboarding", "guide"]
        },
        {
            "title": "Sprint Retrospective - Week 1",
            "content": "## Sprint Goals\n- Complete user authentication\n- Finish landing page design\n\n## What went well\n- Good collaboration\n- Met most deadlines",
            "author_id": user_ids[1],
            "tags": ["retrospective", "sprint"]
        }
    ]
    
    for page_data in wiki_pages:
        page = WikiPage(**page_data)
        await db.wiki_pages.insert_one(page.dict())
    
    # Update user statistics and calculate burnout risk
    for user_id in user_ids:
        completed_tasks = await db.tasks.count_documents({
            "$or": [
                {"assigned_to": user_id},
                {"assigned_users": {"$in": [user_id]}}
            ],
            "status": TaskStatus.DONE
        })
        
        total_hours = sum([
            entry["hours"] for entry in await db.time_entries.find({
                "user_id": user_id
            }).to_list(1000)
        ])
        
        productivity_score = (completed_tasks * 10) + (total_hours * 0.5)
        burnout_risk = await calculate_burnout_risk(user_id)
        
        await db.users.update_one(
            {"id": user_id},
            {
                "$set": {
                    "total_tasks_completed": completed_tasks,
                    "total_hours_logged": total_hours,
                    "productivity_score": productivity_score,
                    "burnout_risk": burnout_risk
                }
            }
        )
        
        # Update badges
        await update_user_badges(user_id)
    
    return {"message": "Enhanced sample data initialized successfully"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()