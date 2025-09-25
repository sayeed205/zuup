# API Documentation

This document describes the REST API endpoints for Zuup.

## Base URL

```
http://localhost:8080/api/v1
```

## Authentication

Currently, no authentication is required for local access.

## Endpoints

### Tasks

- `GET /tasks` - List all download tasks
- `POST /tasks` - Create a new download task
- `GET /tasks/{task_id}` - Get task details
- `PUT /tasks/{task_id}` - Update task configuration
- `DELETE /tasks/{task_id}` - Cancel and remove task
- `POST /tasks/{task_id}/pause` - Pause task
- `POST /tasks/{task_id}/resume` - Resume task

### Configuration

- `GET /config` - Get global configuration
- `PUT /config` - Update global configuration
- `GET /config/tasks/{task_id}` - Get task-specific configuration
- `PUT /config/tasks/{task_id}` - Update task-specific configuration

### System

- `GET /system/info` - Get system information
- `GET /system/stats` - Get download statistics
- `GET /health` - Health check endpoint

## Data Models

Documentation for request/response schemas will be added in task 2.
