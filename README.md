## What is MLOps?

Machine Learning Operations is the process deploying, monitoring, and updating  models in a production environment. Stated simply:

1. Deploy models quickly
2. Easily compare performance to earlier versions

## Before you deploy

Your model is a piece of a larger system. You need to consider how things fit together and how your model is going to be used.

**Do you have data transformation steps before any prediction is made?** You'll need to replicate these steps in an automated way, either through a feature-engineering pipeline or custom code included with your deployed model.

**Have you tested things out locally?** Try moving your completed model to a stand-alone function and test it out with [holdout data](https://www.datarobot.com/wiki/training-validation-holdout/).

## Common approaches

There are many ways that models can be moved into production.

If you're doing offline batch predictions, you may only need to upload your code and model objects to an accessible location within your system (e.g. AWS S3, Azure Storage, GCP Bucket). Then, in whatever pipeline you're using to make those predictions, you can load your model from that storage location and you're good to go.

For online predictions, you'll need to host your model at a URL endpoint. This will allow users to send requests and make predictions in real-time. **For our comparison, we'll focus on making online predictions**.

## Method 1: Cloud Functions

![Notion_GCdieJqk5N](https://user-images.githubusercontent.com/109352381/201939679-3509c1d1-23f3-47d0-8872-d57ea420d41e.png)

Cloud functions are one of the simplest ways to create a basic model endpoint. Examples would be *AWS Lambda*, *Azure Functions*, *Google Cloud Functions*.

**Basic steps:** 

1. Create a web API 
2. Create a route that accepts user data, loads your model, and returns predictions 
3. Upload your files to a cloud storage location 
4. Create a cloud function that references your cloud storage location

**Pros:** 

- Fewest steps to deploy
- Auto-scales to handle requests at scale
- Typically the least expensive way to serve a model endpoint

**Cons:** 

- Limited ability to configure runtime environment
- Limited to a single URL endpoint (one serving function)

## Method 2: Cloud Container Functions

![Notion_ZvCAPhAUB5](https://user-images.githubusercontent.com/109352381/201939813-d5340dc3-fda5-4904-bc33-2bc78c0aa9e6.png)

Cloud container functions are very similar to normal cloud functions, the main difference being that you're deploying a custom image of your runtime, rather than using managed one. Examples would be *AWS App Runner*, *Azure Container Apps*, *Google Cloud Run*.

**Basic steps:** 

1. Create a web API 
2. Create a route that accepts user data, loads your model, and returns predictions 
3. Create a Dockerfile that will run your app 
4. Build your Docker image and push it to your platform's image repo 
5. Deploy your image

**Pros:** 

- Auto-scales to handle requests at scale
- Pay only for compute needed to process the request
- Fully customizable environment via Docker
- Create as many URL endpoints as you want

**Cons:** 

- More development overhead, as you're responsible for building and maintaining a container for your application

## Method 3: Model Hosting Service

![Notion_EPFEyngpUW](https://user-images.githubusercontent.com/109352381/201939905-d6f8b1b5-a865-4512-96b6-c960516c9414.png)


Finally, we get to managed machine learning services. These services typically offer the ability to train, evaluate, and serve models for other applications to consume. In our example we'll focus just on model serving. Example services that you can use would be *AWS SageMaker*, *Azure ML Service*, *Google Vertex AI*.

**Basic steps:** 

1. Upload your trained model file to a cloud container 
2. Register your model and create an endpoint URL through the managed ML services mentioned above

**Pros:** 

- Auto-scales to handle requests at scale
- Pay only for compute needed to process the request
- Highly optimized for different model architectures

**Cons:** 

- Complex deployment process
- Limited to a single URL endpoint (one model per endpoint)

## Performance

![Notion_DbQjQZ1uOD](https://user-images.githubusercontent.com/109352381/201939962-4c389ee8-e93d-4e4c-bd18-d0cc13bd2572.png)

For this test, I measured the response time for each model+deployment combination. Each endpoint is pinged 200 times, with 10 requests being made concurrently. **In almost all cases, the average response time was nearly indistinguishable**.

Not surprisingly, the ML Hosting Services performed the best. What is interesting is that the ML Service Hosted **deep neural net model actually out-performed a simple "hello world" response** from our Flask server.

## So, which method should you use?

**Cloud Functions:** I feel like these are slowly being phased out in favor of Cloud Container Functions. If you are already familiar with cloud functions, and you prefer to outsource your concerns about where your code is running, this is still a fine solution. The ease of deployment, scalability, and performance should still suffice for most applications.

**Cloud Container Functions:** This is my default preference. If you're already familiar with Docker, it's very little additional work to deploy your container. With the ability to scale-up and scale-down on demand, in addition to only paying for the compute you use, you have a cost-efficient and highly-configurable way to deploy your models (or any other app of your pleasing).

**Managed ML Services:**  The benefits of using ML managed services come in two flavors:

- First, you're working in an opinionated environment that enforces basic MLOps considerations. You'll learn basic concepts by going through the required steps to deploy your models. For example, how to standardize your model training process, how to handle model versions, and you get ready-made model monitoring & evaluation tools.
- Second, pure speed. You'll be hard-pressed to out-perform these highly optimized runtime environments.
