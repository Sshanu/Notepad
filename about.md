---
layout: page
permalink: /about/index.html
title: Shanu Kumar
tags: [Shanu, Kumar]
imagefeature: fourseasons.jpg
chart: true

---
<style type="text/css">
	.img-circle {
    border-radius: 50%;
}
</style>
<figure>
  <img src="{{ site.url }}/images/shanu.jpg" alt="Shanu Kumar" class="img-circle" height="300" width="300">
</figure>

{% assign total_words = 0 %}
{% assign total_readtime = 0 %}
{% assign featuredcount = 0 %}
{% assign statuscount = 0 %}

{% for post in site.posts %}
    {% assign post_words = post.content | strip_html | number_of_words %}
    {% assign readtime = post_words | append: '.0' | divided_by:200 %}
    {% assign total_words = total_words | plus: post_words %}
    {% assign total_readtime = total_readtime | plus: readtime %}
    {% if post.featured %}
    {% assign featuredcount = featuredcount | plus: 1 %}
    {% endif %}
{% endfor %}


I am 3rd year undergrad in **Electrical Engineering** at **IIT Kanpur** who loves to code, read and watch cricket.

